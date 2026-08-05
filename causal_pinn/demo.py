"""Visualization-oriented causal PINN reconstruction demo using only stdlib."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, pi, sin, sqrt
from pathlib import Path
import random


@dataclass(frozen=True)
class ReconstructionResult:
    """Container for a reconstructed field and its evaluation metrics."""

    name: str
    field: list[list[float]]
    rmse: float
    relative_l2: float
    physics_residual: float


def generate_heat_equation_data(
    nx: int = 64,
    nt: int = 50,
    diffusivity: float = 0.08,
    noise_std: float = 0.02,
    observation_fraction: float = 0.18,
    seed: int = 7,
) -> dict[str, object]:
    """Generate synthetic heat-equation data and sparse observations."""

    rng = random.Random(seed)
    x = [i / (nx - 1) for i in range(nx)]
    t = [i / (nt - 1) for i in range(nt)]
    truth = [
        [
            sin(pi * xi) * exp(-diffusivity * pi**2 * ti)
            + 0.45 * sin(3 * pi * xi) * exp(-diffusivity * (3 * pi) ** 2 * ti)
            for xi in x
        ]
        for ti in t
    ]
    observations = [[value + rng.gauss(0.0, noise_std) for value in row] for row in truth]
    mask = [[rng.random() < observation_fraction for _ in range(nx)] for _ in range(nt)]
    for xi in range(nx):
        mask[0][xi] = True
    for ti in range(nt):
        mask[ti][0] = True
        mask[ti][-1] = True
    return {"x": x, "t": t, "truth": truth, "observations": observations, "mask": mask, "diffusivity": diffusivity}


def _interpolate_initial(observations: list[list[float]], mask: list[list[bool]]) -> list[list[float]]:
    field = [row[:] for row in observations]
    nt, nx = len(field), len(field[0])
    for ti in range(nt):
        known = [xi for xi, seen in enumerate(mask[ti]) if seen]
        for xi in range(nx):
            if mask[ti][xi]:
                continue
            left = max((k for k in known if k < xi), default=known[0])
            right = min((k for k in known if k > xi), default=known[-1])
            if left == right:
                field[ti][xi] = observations[ti][left]
            else:
                alpha = (xi - left) / (right - left)
                field[ti][xi] = (1 - alpha) * observations[ti][left] + alpha * observations[ti][right]
    return field


def _physics_step(field: list[list[float]], diffusivity: float, dt: float, dx: float) -> list[list[float]]:
    predicted = [field[0][:]]
    coeff = min(0.45, diffusivity * dt / (dx * dx))
    for ti in range(len(field) - 1):
        row = field[ti][:]
        nxt = row[:]
        for xi in range(1, len(row) - 1):
            nxt[xi] = row[xi] + coeff * (row[xi - 1] - 2.0 * row[xi] + row[xi + 1])
        nxt[0] = 0.0
        nxt[-1] = 0.0
        predicted.append(nxt)
    return predicted


def reconstruct_field(
    observations: list[list[float]],
    mask: list[list[bool]],
    truth: list[list[float]],
    diffusivity: float,
    name: str,
    physics_weight: float = 0.0,
    causal: bool = False,
    iterations: int = 180,
) -> ReconstructionResult:
    """Reconstruct a field from sparse observations with optional PINN-style terms."""

    nt, nx = len(observations), len(observations[0])
    field = _interpolate_initial(observations, mask)
    dt, dx = 1.0 / (nt - 1), 1.0 / (nx - 1)
    for _ in range(iterations):
        physics = _physics_step(field, diffusivity, dt, dx) if physics_weight else field
        updated = [row[:] for row in field]
        for ti in range(nt):
            causal_weight = 1.0 if not causal else 1.0 + 2.5 * (1.0 - ti / (nt - 1))
            for xi in range(1, nx - 1):
                smooth = (field[ti][xi - 1] + field[ti][xi + 1]) / 2.0
                value = 0.82 * field[ti][xi] + 0.18 * smooth
                if physics_weight:
                    value = (1.0 - physics_weight) * value + physics_weight * physics[ti][xi]
                if mask[ti][xi]:
                    obs_blend = min(0.92, 0.35 * causal_weight)
                    value = (1.0 - obs_blend) * value + obs_blend * observations[ti][xi]
                updated[ti][xi] = value
            updated[ti][0] = 0.0
            updated[ti][-1] = 0.0
        field = updated
    return ReconstructionResult(name, field, _rmse(field, truth), _relative_l2(field, truth), _physics_residual(field, diffusivity))


def _rmse(field: list[list[float]], truth: list[list[float]]) -> float:
    values = [(a - b) ** 2 for row_a, row_b in zip(field, truth) for a, b in zip(row_a, row_b)]
    return sqrt(sum(values) / len(values))


def _relative_l2(field: list[list[float]], truth: list[list[float]]) -> float:
    num = sum((a - b) ** 2 for row_a, row_b in zip(field, truth) for a, b in zip(row_a, row_b))
    den = sum(b * b for row in truth for b in row)
    return sqrt(num / den)


def _physics_residual(field: list[list[float]], diffusivity: float) -> float:
    nt, nx = len(field), len(field[0])
    dt, dx = 1.0 / (nt - 1), 1.0 / (nx - 1)
    residuals = []
    for ti in range(nt - 1):
        for xi in range(1, nx - 1):
            ut = (field[ti + 1][xi] - field[ti][xi]) / dt
            uxx = (field[ti][xi - 1] - 2 * field[ti][xi] + field[ti][xi + 1]) / (dx * dx)
            residuals.append((ut - diffusivity * uxx) ** 2)
    return sqrt(sum(residuals) / len(residuals))


def build_comparison(seed: int = 7) -> tuple[dict[str, object], list[ReconstructionResult]]:
    """Build all reconstruction variants used in the comparison plot."""

    data = generate_heat_equation_data(seed=seed)
    kwargs = {"observations": data["observations"], "mask": data["mask"], "truth": data["truth"], "diffusivity": data["diffusivity"]}
    return data, [
        reconstruct_field(**kwargs, name="Data only", physics_weight=0.0, causal=False),
        reconstruct_field(**kwargs, name="Physics constrained", physics_weight=0.01, causal=False),
        reconstruct_field(**kwargs, name="Causal data", physics_weight=0.0, causal=True),
        reconstruct_field(**kwargs, name="Causal PINN", physics_weight=0.01, causal=True),
    ]


def _color(value: float, vmin: float, vmax: float) -> str:
    z = 0.0 if vmax == vmin else max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    r = int(68 + 185 * z)
    g = int(1 + 230 * (1 - abs(z - 0.55) / 0.55))
    b = int(84 + 70 * (1 - z))
    return f"#{r:02x}{max(0, min(255, g)):02x}{b:02x}"


def _heatmap_svg(field: list[list[float | None]], x0: int, y0: int, width: int, height: int, vmin: float, vmax: float) -> str:
    nt, nx = len(field), len(field[0])
    cw, ch = width / nx, height / nt
    parts = []
    for ti, row in enumerate(field):
        for xi, value in enumerate(row):
            fill = "#f1f1f1" if value is None else _color(float(value), vmin, vmax)
            parts.append(f'<rect x="{x0 + xi*cw:.2f}" y="{y0 + ti*ch:.2f}" width="{cw+0.2:.2f}" height="{ch+0.2:.2f}" fill="{fill}"/>')
    return "\n".join(parts)


def plot_comparison(data: dict[str, object], results: list[ReconstructionResult], output_path: str | Path = "outputs/causal_pinn_comparison.svg") -> Path:
    """Plot data, reconstructions, and quantitative performance comparison as SVG."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    truth = data["truth"]
    observations = data["observations"]
    mask = data["mask"]
    observed = [[observations[t][x] if mask[t][x] else None for x in range(len(mask[0]))] for t in range(len(mask))]
    vmin, vmax = min(min(row) for row in truth), max(max(row) for row in truth)
    panels = [("Ground truth", truth), ("Sparse noisy data", observed)] + [(r.name, r.field) for r in results]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">', '<rect width="100%" height="100%" fill="white"/>', '<text x="700" y="35" text-anchor="middle" font-size="28" font-family="Arial">Causal PINN reconstruction comparison</text>']
    for i, (title, field) in enumerate(panels):
        col, row = i % 3, i // 3
        x0, y0 = 50 + col * 430, 75 + row * 300
        svg.append(f'<text x="{x0 + 180}" y="{y0 - 12}" text-anchor="middle" font-size="18" font-family="Arial">{title}</text>')
        svg.append(_heatmap_svg(field, x0, y0, 360, 220, vmin, vmax))
        svg.append(f'<rect x="{x0}" y="{y0}" width="360" height="220" fill="none" stroke="#333"/>')
        svg.append(f'<text x="{x0 + 180}" y="{y0 + 250}" text-anchor="middle" font-size="13" font-family="Arial">x</text>')
        svg.append(f'<text x="{x0 - 25}" y="{y0 + 110}" text-anchor="middle" transform="rotate(-90 {x0 - 25} {y0 + 110})" font-size="13" font-family="Arial">t</text>')
    chart_x, chart_y = 80, 700
    max_metric = max(max(r.rmse, r.relative_l2) for r in results)
    svg.append('<text x="310" y="665" text-anchor="middle" font-size="18" font-family="Arial">Reconstruction performance</text>')
    for i, r in enumerate(results):
        base = chart_x + i * 115
        for j, (value, color) in enumerate(((r.rmse, "#1b9e77"), (r.relative_l2, "#7570b3"))):
            h = 140 * value / max_metric
            svg.append(f'<rect x="{base + j*32}" y="{chart_y + 150 - h:.2f}" width="28" height="{h:.2f}" fill="{color}"/>')
        svg.append(f'<text x="{base + 30}" y="{chart_y + 175}" text-anchor="middle" font-size="11" font-family="Arial" transform="rotate(25 {base + 30} {chart_y + 175})">{r.name}</text>')
    svg.append('<text x="565" y="705" font-size="12" font-family="Arial" fill="#1b9e77">RMSE</text><text x="565" y="725" font-size="12" font-family="Arial" fill="#7570b3">Relative L2</text>')
    svg.append('<text x="930" y="665" text-anchor="middle" font-size="18" font-family="Arial">Physics residual (lower is better)</text>')
    max_res = max(r.physics_residual for r in results)
    for i, r in enumerate(results):
        base = 720 + i * 130
        h = 145 * r.physics_residual / max_res
        svg.append(f'<rect x="{base}" y="{chart_y + 150 - h:.2f}" width="55" height="{h:.2f}" fill="#d95f02"/>')
        svg.append(f'<text x="{base + 28}" y="{chart_y + 175}" text-anchor="middle" font-size="11" font-family="Arial" transform="rotate(25 {base + 28} {chart_y + 175})">{r.name}</text>')
    svg.append('</svg>')
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the causal PINN demo and write the comparison SVG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/causal_pinn_comparison.svg"),
        help="Path for the generated SVG comparison plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used to generate sparse noisy observations.",
    )
    return parser


def run_application(
    seed: int = 7,
    output_path: str | Path = "outputs/causal_pinn_comparison.svg",
) -> Path:
    """Run the full causal-PINN data-generation, reconstruction, and plotting app."""

    data, results = build_comparison(seed=seed)
    output = plot_comparison(data, results, output_path)
    print(f"Wrote {output}")
    for result in results:
        print(
            f"{result.name}: RMSE={result.rmse:.4f}, "
            f"Relative L2={result.relative_l2:.4f}, "
            f"Physics residual={result.physics_residual:.4e}"
        )
    return output


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for running the complete causal PINN application."""

    args = _build_parser().parse_args(argv)
    run_application(seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
