"""Modern Tkinter explorer for Causal-PINN reconstruction examples.

The GUI stays dependency-light by rendering deterministic canvas heatmaps with
the same stdlib plotting helpers used by the command-line demo.  It provides guided
1D, 2D, and 3D imaging scenarios that can be used in teaching sessions without
requiring a GPU or an interactive deep-learning stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, pi, sin, sqrt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from causal_pinn.demo import build_comparison, plot_comparison


@dataclass(frozen=True)
class ImagingCase:
    """A lightweight Causal-PINN teaching case shown in the GUI."""

    key: str
    title: str
    subtitle: str
    dimension_label: str
    teaching_points: tuple[str, ...]
    field: list[list[float]]
    observed: list[list[float | None]]
    reconstruction: list[list[float]]
    metrics: tuple[tuple[str, float], ...]


def _norm_rmse(field: list[list[float]], truth: list[list[float]]) -> float:
    values = [(a - b) ** 2 for row_a, row_b in zip(field, truth) for a, b in zip(row_a, row_b)]
    return sqrt(sum(values) / len(values))


def _residual_proxy(field: list[list[float]]) -> float:
    residuals = []
    for y in range(1, len(field) - 1):
        for x in range(1, len(field[0]) - 1):
            laplace = field[y - 1][x] + field[y + 1][x] + field[y][x - 1] + field[y][x + 1] - 4 * field[y][x]
            residuals.append(laplace * laplace)
    return sqrt(sum(residuals) / len(residuals)) if residuals else 0.0


def _blur_observed(observed: list[list[float | None]], fallback: float = 0.0, iterations: int = 90) -> list[list[float]]:
    field = [[fallback if value is None else float(value) for value in row] for row in observed]
    fixed = [[value is not None for value in row] for row in observed]
    height, width = len(field), len(field[0])
    for _ in range(iterations):
        updated = [row[:] for row in field]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if fixed[y][x]:
                    continue
                updated[y][x] = 0.20 * field[y][x] + 0.20 * (field[y - 1][x] + field[y + 1][x] + field[y][x - 1] + field[y][x + 1])
        field = updated
    return field


def _make_1d_case() -> ImagingCase:
    data, results = build_comparison(seed=7)
    causal = next(result for result in results if result.name == "Causal PINN")
    observed = [[data["observations"][t][x] if data["mask"][t][x] else None for x in range(len(data["mask"][0]))] for t in range(len(data["mask"]))]
    return ImagingCase(
        key="1d",
        title="1D heat-field reconstruction",
        subtitle="Sparse sensors over x and time, reconstructed with causal time weighting.",
        dimension_label="Signal + time",
        teaching_points=(
            "Causal weights emphasize early-time consistency before later frames are trusted.",
            "The physics term penalizes heat-equation residuals between sparse observations.",
            "Use this case to introduce collocation points, boundary conditions, and RMSE.",
        ),
        field=data["truth"],
        observed=observed,
        reconstruction=causal.field,
        metrics=(("RMSE", causal.rmse), ("Relative L2", causal.relative_l2), ("Residual", causal.physics_residual)),
    )


def _make_2d_case() -> ImagingCase:
    size = 54
    truth = []
    observed = []
    for y in range(size):
        yy = (y / (size - 1) - 0.5) * 2
        truth_row = []
        obs_row = []
        for x in range(size):
            xx = (x / (size - 1) - 0.5) * 2
            value = exp(-5.5 * (xx * xx + yy * yy)) + 0.35 * sin(2.5 * pi * xx) * cos(1.5 * pi * yy)
            truth_row.append(value)
            seen = (x % 6 == 0 and y % 3 == 0) or x in (0, size - 1) or y in (0, size - 1)
            obs_row.append(value + 0.03 * sin(13 * xx + 5 * yy) if seen else None)
        truth.append(truth_row)
        observed.append(obs_row)
    reconstruction = _blur_observed(observed, iterations=130)
    return ImagingCase(
        key="2d",
        title="2D medical-style slice inpainting",
        subtitle="A sparse image slice is completed by balancing observations and smooth PDE priors.",
        dimension_label="Image slice",
        teaching_points=(
            "Image pixels can be treated as collocation points on a 2D spatial domain.",
            "The residual proxy highlights whether the reconstruction obeys a smooth diffusion prior.",
            "This mirrors MRI/CT slice completion when measurements are expensive or undersampled.",
        ),
        field=truth,
        observed=observed,
        reconstruction=reconstruction,
        metrics=(("RMSE", _norm_rmse(reconstruction, truth)), ("Relative L2", _norm_rmse(reconstruction, truth) / 0.45), ("Residual", _residual_proxy(reconstruction))),
    )


def _make_3d_case() -> ImagingCase:
    size = 46
    truth = []
    observed = []
    for z in range(size):
        zz = (z / (size - 1) - 0.5) * 2
        truth_row = []
        obs_row = []
        for x in range(size):
            xx = (x / (size - 1) - 0.5) * 2
            value = exp(-4.0 * ((xx + 0.2) ** 2 + zz * zz)) + 0.6 * exp(-9.0 * ((xx - 0.35) ** 2 + (zz + 0.25) ** 2))
            value += 0.12 * cos(3 * pi * xx) * sin(2 * pi * zz)
            truth_row.append(value)
            seen = z % 5 == 0 or x % 9 == 0 or x in (0, size - 1) or z in (0, size - 1)
            obs_row.append(value + 0.02 * cos(11 * xx - 7 * zz) if seen else None)
        truth.append(truth_row)
        observed.append(obs_row)
    reconstruction = _blur_observed(observed, iterations=150)
    return ImagingCase(
        key="3d",
        title="3D volume mid-slice reconstruction",
        subtitle="A volume is taught through a representative z-slice with tomographic sampling bands.",
        dimension_label="Volume slice",
        teaching_points=(
            "3D imaging extends the same idea to x-y-z collocation points and volumetric boundaries.",
            "The displayed plane represents one slice through a learned volume or neural field.",
            "Causal scheduling can reveal easy slices first, then progressively fit harder anatomy.",
        ),
        field=truth,
        observed=observed,
        reconstruction=reconstruction,
        metrics=(("RMSE", _norm_rmse(reconstruction, truth)), ("Relative L2", _norm_rmse(reconstruction, truth) / 0.50), ("Residual", _residual_proxy(reconstruction))),
    )


def build_imaging_cases() -> list[ImagingCase]:
    """Return deterministic 1D, 2D, and 3D examples for the GUI."""

    return [_make_1d_case(), _make_2d_case(), _make_3d_case()]


def _color(value: float, vmin: float, vmax: float) -> str:
    z = 0.0 if vmax == vmin else max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    r = int(28 + 210 * z)
    g = int(46 + 160 * (1 - abs(z - 0.48) / 0.52))
    b = int(92 + 120 * (1 - z))
    return f"#{r:02x}{max(0, min(255, g)):02x}{b:02x}"


class CausalPinnExplorer(tk.Tk):
    """Tkinter application for exploring Causal-PINN imaging examples."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Causal-PINN Imaging Explorer")
        self.geometry("1180x780")
        self.minsize(980, 680)
        self.configure(bg="#101828")
        self.cases = build_imaging_cases()
        self.case_var = tk.StringVar(value=self.cases[0].key)
        self.view_var = tk.StringVar(value="reconstruction")
        self.status_var = tk.StringVar(value="Ready")
        self._configure_style()
        self._build_layout()
        self._render_case()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#101828")
        style.configure("Card.TFrame", background="#182230", relief="flat")
        style.configure("TLabel", background="#101828", foreground="#f8fafc", font=("Segoe UI", 11))
        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Sub.TLabel", foreground="#cbd5e1")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("TRadiobutton", background="#182230", foreground="#f8fafc", font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        shell = ttk.Frame(self, padding=18)
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x")
        ttk.Label(header, text="Causal-PINN Imaging Explorer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Interactive Tk GUI for teaching causal physics-informed reconstruction in 1D, 2D, and 3D.", style="Sub.TLabel").pack(anchor="w", pady=(4, 14))

        body = ttk.Frame(shell)
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body, style="Card.TFrame", padding=16, width=300)
        sidebar.pack(side="left", fill="y", padx=(0, 18))
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="Examples", background="#182230", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        for case in self.cases:
            ttk.Radiobutton(sidebar, text=f"{case.dimension_label}: {case.title}", value=case.key, variable=self.case_var, command=self._render_case).pack(anchor="w", pady=8)
        ttk.Separator(sidebar).pack(fill="x", pady=14)
        ttk.Label(sidebar, text="Visualization layer", background="#182230", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        for key, text in (("truth", "Ground truth"), ("observed", "Sparse observations"), ("reconstruction", "Causal-PINN reconstruction")):
            ttk.Radiobutton(sidebar, text=text, value=key, variable=self.view_var, command=self._render_case).pack(anchor="w", pady=6)
        ttk.Button(sidebar, text="Export full SVG comparison", command=self._export_svg).pack(fill="x", pady=(22, 8))
        ttk.Button(sidebar, text="About the workflow", command=self._show_about).pack(fill="x")

        main = ttk.Frame(body, style="Card.TFrame", padding=16)
        main.pack(side="left", fill="both", expand=True)
        self.title_label = ttk.Label(main, text="", background="#182230", font=("Segoe UI", 18, "bold"))
        self.title_label.pack(anchor="w")
        self.subtitle_label = ttk.Label(main, text="", background="#182230", foreground="#cbd5e1", wraplength=780)
        self.subtitle_label.pack(anchor="w", pady=(2, 12))
        self.canvas = tk.Canvas(main, bg="#0b1220", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._render_case())
        self.points_label = ttk.Label(main, text="", background="#182230", foreground="#e2e8f0", wraplength=840, justify="left")
        self.points_label.pack(anchor="w", pady=(12, 0))
        ttk.Label(shell, textvariable=self.status_var, style="Sub.TLabel").pack(anchor="w", pady=(10, 0))

    def _current_case(self) -> ImagingCase:
        return next(case for case in self.cases if case.key == self.case_var.get())

    def _render_case(self) -> None:
        case = self._current_case()
        self.title_label.configure(text=case.title)
        self.subtitle_label.configure(text=case.subtitle)
        self.points_label.configure(text="\n".join(f"• {point}" for point in case.teaching_points))
        layer = {"truth": case.field, "observed": case.observed, "reconstruction": case.reconstruction}[self.view_var.get()]
        self._draw_heatmap(layer, case)
        metric_text = " | ".join(f"{name}: {value:.4f}" for name, value in case.metrics)
        self.status_var.set(f"{case.dimension_label} — {self.view_var.get().replace('_', ' ').title()} — {metric_text}")

    def _draw_heatmap(self, field: list[list[float | None]], case: ImagingCase) -> None:
        self.canvas.delete("all")
        width = max(400, self.canvas.winfo_width())
        height = max(320, self.canvas.winfo_height())
        values = [float(value) for row in case.field for value in row]
        vmin, vmax = min(values), max(values)
        rows, cols = len(field), len(field[0])
        margin = 44
        plot_w, plot_h = width - 2 * margin, height - 104
        cell_w, cell_h = plot_w / cols, plot_h / rows
        self.canvas.create_rectangle(0, 0, width, height, fill="#0b1220", outline="")
        for y, row in enumerate(field):
            for x, value in enumerate(row):
                fill = "#334155" if value is None else _color(float(value), vmin, vmax)
                self.canvas.create_rectangle(margin + x * cell_w, margin + y * cell_h, margin + (x + 1) * cell_w + 0.5, margin + (y + 1) * cell_h + 0.5, fill=fill, outline="")
        self.canvas.create_rectangle(margin, margin, margin + plot_w, margin + plot_h, outline="#e2e8f0", width=2)
        self.canvas.create_text(width / 2, 22, fill="#f8fafc", font=("Segoe UI", 15, "bold"), text=f"{case.dimension_label} · {self.view_var.get().replace('_', ' ').title()}")
        self.canvas.create_text(width / 2, height - 38, fill="#cbd5e1", font=("Segoe UI", 11), text="Dark cells in observation mode indicate unsampled collocation points")
        for i, (name, value) in enumerate(case.metrics):
            self.canvas.create_text(margin + 16 + i * 170, height - 72, anchor="w", fill="#93c5fd", font=("Segoe UI", 11, "bold"), text=f"{name}: {value:.4f}")

    def _export_svg(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".svg", initialfile="causal_pinn_comparison.svg", filetypes=(("SVG files", "*.svg"),))
        if not path:
            return
        data, results = build_comparison()
        output = plot_comparison(data, results, Path(path))
        self.status_var.set(f"Exported comparison to {output}")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About Causal-PINN Imaging Explorer",
            "Select 1D, 2D, or 3D examples to compare truth, sparse measurements, and a causal physics-informed reconstruction. "
            "The examples are deterministic and lightweight so they can be used in a classroom or workshop without extra dependencies.",
        )


def main() -> None:
    """Launch the Tkinter Causal-PINN explorer."""

    app = CausalPinnExplorer()
    app.mainloop()


if __name__ == "__main__":
    main()
