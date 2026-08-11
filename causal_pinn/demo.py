"""Teaching demos for causal and Bayesian PINN-style infrared inverse problems.

The module intentionally uses only the Python standard library so the examples
can run in lightweight classrooms, notebooks, and CI jobs without GPU or deep
learning dependencies. The numerical routines are compact surrogates for PINN
training loops: sparse data terms, physics residual terms, causal time weights,
and Bayesian model averaging are all exposed in code that students can read in a
single sitting.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class BayesianInfraredResult:
    """Posterior summary for a Bayesian PINN-style infrared inversion."""

    posterior_mean: list[list[float]]
    posterior_std: list[list[float]]
    diffusivity_grid: list[float]
    posterior_weights: list[float]
    map_diffusivity: float
    emissivity_estimate: float
    temperature_rmse: float
    credible_coverage: float


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


def generate_infrared_inverse_data(
    nx: int = 64,
    nt: int = 48,
    diffusivity: float = 0.075,
    emissivity: float = 0.91,
    ambient_temperature: float = 293.15,
    temperature_scale: float = 22.0,
    noise_std: float = 0.018,
    observation_fraction: float = 0.16,
    seed: int = 11,
) -> dict[str, object]:
    """Generate sparse infrared radiance observations for an inverse heat problem.

    The hidden state is a transient one-dimensional surface temperature field.
    A simplified infrared camera observes normalized radiance proportional to
    emissivity times temperature contrast. The inverse task is to infer both the
    temperature field and unknown material diffusivity from sparse, noisy pixels.
    """

    heat = generate_heat_equation_data(
        nx=nx,
        nt=nt,
        diffusivity=diffusivity,
        noise_std=0.0,
        observation_fraction=observation_fraction,
        seed=seed,
    )
    rng = random.Random(seed + 101)
    temperature = [
        [ambient_temperature + temperature_scale * value for value in row]
        for row in heat["truth"]
    ]
    normalized_truth = [[(value - ambient_temperature) / temperature_scale for value in row] for row in temperature]
    radiance = [[emissivity * value + rng.gauss(0.0, noise_std) for value in row] for row in normalized_truth]
    return {
        "x": heat["x"],
        "t": heat["t"],
        "truth": normalized_truth,
        "temperature_truth": temperature,
        "radiance": radiance,
        "mask": heat["mask"],
        "diffusivity": diffusivity,
        "emissivity": emissivity,
        "ambient_temperature": ambient_temperature,
        "temperature_scale": temperature_scale,
        "noise_std": noise_std,
    }


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


def infer_bayesian_infrared_pinn(data: dict[str, object], diffusivity_grid: list[float] | None = None) -> BayesianInfraredResult:
    """Infer temperature and diffusivity with Bayesian model averaging.

    Each candidate diffusivity acts like a small physics-informed model. Its
    likelihood combines sparse infrared data mismatch and a heat-equation
    residual, yielding posterior weights that are easy to visualize in class.
    """

    radiance = data["radiance"]
    mask = data["mask"]
    truth = data["truth"]
    true_emissivity = float(data["emissivity"])
    noise_std = float(data["noise_std"])
    diffusivity_grid = diffusivity_grid or [0.04, 0.055, 0.07, 0.085, 0.1, 0.115]
    observed_pairs = [(radiance[t][x], truth[t][x]) for t in range(len(mask)) for x in range(len(mask[0])) if mask[t][x] and abs(truth[t][x]) > 1e-6]
    emissivity_estimate = sum(y * u for y, u in observed_pairs) / sum(u * u for _, u in observed_pairs)
    normalized_observations = [[value / max(emissivity_estimate, 1e-6) for value in row] for row in radiance]

    candidates: list[list[list[float]]] = []
    log_weights: list[float] = []
    for diffusivity in diffusivity_grid:
        result = reconstruct_field(
            normalized_observations,
            mask,
            truth,
            diffusivity,
            name=f"alpha={diffusivity:.3f}",
            physics_weight=0.012,
            causal=True,
            iterations=140,
        )
        candidates.append(result.field)
        data_misfit = sum(
            (emissivity_estimate * result.field[t][x] - radiance[t][x]) ** 2
            for t in range(len(mask))
            for x in range(len(mask[0]))
            if mask[t][x]
        )
        prior = -0.5 * ((diffusivity - 0.08) / 0.035) ** 2
        log_weights.append(prior - 0.5 * data_misfit / (noise_std * noise_std) - 0.02 * result.physics_residual)

    max_log = max(log_weights)
    raw = [exp(weight - max_log) for weight in log_weights]
    total = sum(raw)
    weights = [value / total for value in raw]
    nt, nx = len(truth), len(truth[0])
    mean = [[sum(weights[k] * candidates[k][t][x] for k in range(len(candidates))) for x in range(nx)] for t in range(nt)]
    std = [
        [sqrt(sum(weights[k] * (candidates[k][t][x] - mean[t][x]) ** 2 for k in range(len(candidates))) + noise_std**2) for x in range(nx)]
        for t in range(nt)
    ]
    coverage = sum(
        1 for t in range(nt) for x in range(nx) if abs(truth[t][x] - mean[t][x]) <= 2.0 * std[t][x]
    ) / (nt * nx)
    return BayesianInfraredResult(
        posterior_mean=mean,
        posterior_std=std,
        diffusivity_grid=diffusivity_grid,
        posterior_weights=weights,
        map_diffusivity=diffusivity_grid[weights.index(max(weights))],
        emissivity_estimate=emissivity_estimate if observed_pairs else true_emissivity,
        temperature_rmse=_rmse(mean, truth),
        credible_coverage=coverage,
    )


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


def build_bayesian_infrared_demo(seed: int = 11) -> tuple[dict[str, object], BayesianInfraredResult]:
    """Build the Bayesian PINN inverse-problem teaching example."""

    data = generate_infrared_inverse_data(seed=seed)
    return data, infer_bayesian_infrared_pinn(data)


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


def plot_bayesian_infrared_demo(data: dict[str, object], result: BayesianInfraredResult, output_path: str | Path = "outputs/bayesian_infrared_pinn.svg") -> Path:
    """Plot the Bayesian infrared inverse-problem summary as SVG."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    truth = data["truth"]
    radiance = data["radiance"]
    mask = data["mask"]
    observed = [[radiance[t][x] if mask[t][x] else None for x in range(len(mask[0]))] for t in range(len(mask))]
    vmin, vmax = min(min(row) for row in truth), max(max(row) for row in truth)
    smax = max(max(row) for row in result.posterior_std)
    panels = [
        ("True temperature contrast", truth, vmin, vmax),
        ("Sparse infrared radiance", observed, vmin, vmax),
        ("Posterior mean temperature", result.posterior_mean, vmin, vmax),
        ("Posterior uncertainty", result.posterior_std, 0.0, smax),
    ]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" viewBox="0 0 1200 820">', '<rect width="100%" height="100%" fill="#fbfbfd"/>', '<text x="600" y="38" text-anchor="middle" font-size="28" font-family="Arial">Bayesian PINN for infrared inverse imaging</text>']
    for i, (title, field, lo, hi) in enumerate(panels):
        col, row = i % 2, i // 2
        x0, y0 = 70 + col * 550, 85 + row * 280
        svg.append(f'<text x="{x0 + 220}" y="{y0 - 14}" text-anchor="middle" font-size="18" font-family="Arial">{title}</text>')
        svg.append(_heatmap_svg(field, x0, y0, 440, 205, lo, hi))
        svg.append(f'<rect x="{x0}" y="{y0}" width="440" height="205" fill="none" stroke="#2f2f37"/>')
    svg.append('<text x="290" y="650" text-anchor="middle" font-size="18" font-family="Arial">Posterior over thermal diffusivity</text>')
    max_weight = max(result.posterior_weights)
    for i, (alpha, weight) in enumerate(zip(result.diffusivity_grid, result.posterior_weights)):
        x = 95 + i * 70
        h = 135 * weight / max_weight
        svg.append(f'<rect x="{x}" y="{775 - h:.2f}" width="44" height="{h:.2f}" fill="#4c78a8"/>')
        svg.append(f'<text x="{x + 22}" y="795" text-anchor="middle" font-size="11" font-family="Arial">{alpha:.3f}</text>')
    summary = [
        f"MAP diffusivity: {result.map_diffusivity:.3f}",
        f"Estimated emissivity: {result.emissivity_estimate:.3f}",
        f"Temperature RMSE: {result.temperature_rmse:.4f}",
        f"95% coverage: {100 * result.credible_coverage:.1f}%",
    ]
    svg.append('<text x="790" y="650" text-anchor="middle" font-size="18" font-family="Arial">Teaching takeaways</text>')
    for i, line in enumerate(summary):
        svg.append(f'<text x="690" y="690" dy="{i * 28}" font-size="16" font-family="Arial">• {line}</text>')
    svg.append('</svg>')
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def main() -> None:
    """CLI entry point for generating the comparison figures."""

    data, results = build_comparison()
    output = plot_comparison(data, results)
    print(f"Wrote {output}")
    for result in results:
        print(f"{result.name}: RMSE={result.rmse:.4f}, Relative L2={result.relative_l2:.4f}, Physics residual={result.physics_residual:.4e}")
    ir_data, ir_result = build_bayesian_infrared_demo()
    ir_output = plot_bayesian_infrared_demo(ir_data, ir_result)
    print(f"Wrote {ir_output}")
    print(
        "Bayesian IR PINN: "
        f"MAP diffusivity={ir_result.map_diffusivity:.3f}, "
        f"emissivity={ir_result.emissivity_estimate:.3f}, "
        f"RMSE={ir_result.temperature_rmse:.4f}, "
        f"coverage={100 * ir_result.credible_coverage:.1f}%"
    )


if __name__ == "__main__":
    main()
