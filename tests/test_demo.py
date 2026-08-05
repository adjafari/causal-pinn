import math

from causal_pinn.demo import (
    build_comparison,
    generate_heat_equation_data,
    main,
    plot_comparison,
    run_application,
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


def test_run_application_writes_requested_output(tmp_path):
    output = run_application(seed=3, output_path=tmp_path / "app.svg")

    assert output == tmp_path / "app.svg"
    assert output.exists()


def test_main_accepts_direct_application_arguments(tmp_path):
    output = tmp_path / "cli.svg"

    main(["--seed", "4", "--output", str(output)])

    assert output.exists()
