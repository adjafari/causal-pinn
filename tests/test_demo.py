import math

from causal_pinn.demo import build_comparison, generate_heat_equation_data, plot_comparison
from causal_pinn.gui import build_imaging_cases


def test_generate_heat_equation_data_shapes():
    data = generate_heat_equation_data(nx=16, nt=12, seed=1)

    assert len(data["truth"]) == 12
    assert len(data["truth"][0]) == 16
    assert len(data["observations"]) == 12
    assert len(data["mask"][0]) == 16
    assert all(data["mask"][0])


def test_build_comparison_and_plot(tmp_path):
    data, results = build_comparison(seed=2)
    output = plot_comparison(data, results, tmp_path / "comparison.svg")

    assert len(results) == 4
    assert all(math.isfinite(result.rmse) for result in results)
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<svg")


def test_build_imaging_cases():
    cases = build_imaging_cases()

    assert [case.key for case in cases] == ["1d", "2d", "3d"]
    assert all(case.field and case.observed and case.reconstruction for case in cases)
    assert all(len(case.metrics) == 3 for case in cases)
    assert all(math.isfinite(value) for case in cases for _, value in case.metrics)
