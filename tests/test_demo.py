import math

from causal_pinn.demo import (
    build_bayesian_infrared_demo,
    build_comparison,
    generate_heat_equation_data,
    generate_infrared_inverse_data,
    plot_bayesian_infrared_demo,
    plot_comparison,
)


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


def test_generate_infrared_inverse_data_contains_radiance():
    data = generate_infrared_inverse_data(nx=18, nt=14, seed=3)

    assert len(data["temperature_truth"]) == 14
    assert len(data["radiance"][0]) == 18
    assert 0.0 < data["emissivity"] <= 1.0
    assert any(any(row) for row in data["mask"])


def test_build_bayesian_infrared_demo_and_plot(tmp_path):
    data, result = build_bayesian_infrared_demo(seed=4)
    output = plot_bayesian_infrared_demo(data, result, tmp_path / "bayesian_ir.svg")

    assert len(result.diffusivity_grid) == len(result.posterior_weights)
    assert math.isclose(sum(result.posterior_weights), 1.0)
    assert result.map_diffusivity in result.diffusivity_grid
    assert result.temperature_rmse < 0.25
    assert 0.0 <= result.credible_coverage <= 1.0
    assert output.exists()
    assert "Bayesian PINN" in output.read_text(encoding="utf-8")
