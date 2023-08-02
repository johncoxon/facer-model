import numpy as np
import pytest
from birkeland import Model, BetterModel
from datetime import datetime
from pandas import read_csv
from pathlib import Path

model_types = ("model", "north", "south")


@pytest.fixture
def benchmarks():
    benchmarks = {}

    for t in model_types:
        filename = f"test_data_{t}.csv"
        benchmarks[t] = read_csv(Path(__file__).parent / filename)

    return benchmarks


@pytest.fixture
def model_outputs(benchmarks):
    model_outputs = {}
    f_107 = 100

    for t in model_types:
        model_outputs[t] = []

        for cnt, row in benchmarks[t].iterrows():
            if t == "model":
                model_outputs[t].append(Model(row["phi_d"], row["phi_n"], row["f_pc"]))
            else:
                model_outputs[t].append(BetterModel(row["phi_d"], row["phi_n"], f_107,
                                                    datetime(2010, 1, 1), t, f_pc=row["f_pc"]))

        model_outputs[t] = np.array(model_outputs[t])

    return model_outputs


@pytest.mark.parametrize("b_r", (0, 5, 10, 15, 20, 25, 30))
def test_b_r(b_r, benchmarks, model_outputs):
    output = np.array([e.b_r(b_r) for e in model_outputs["model"]])
    assert output == pytest.approx(benchmarks["model"][f"b_r_{b_r}"].values)


@pytest.mark.parametrize("method", ("lambda_r1", "v_r1", "e_b", "e_d", "e_n"))
def test_methods(method, benchmarks, model_outputs):
    output = np.array([getattr(e, method)() for e in model_outputs["model"]])
    assert output == pytest.approx(benchmarks["model"][method].values)


@pytest.mark.parametrize("grid", ("phi", "e_labda", "e_theta", "v_labda", "v_theta", "j"))
def test_model(grid, benchmarks, model_outputs):
    mean_output = np.array([np.mean(np.abs(getattr(e, grid))) for e in model_outputs["model"]])
    sum_output = np.array([np.sum(np.abs(getattr(e, grid))) for e in model_outputs["model"]])

    assert mean_output == pytest.approx(benchmarks["model"][f"{grid}_mean"].values)
    assert sum_output == pytest.approx(benchmarks["model"][f"{grid}_sum"].values)


@pytest.mark.parametrize("grid", ("sza", "sigma_h", "sigma_p", "div_jh", "div_jp"))
@pytest.mark.parametrize("hemisphere", ("north", "south"))
def test_better_model(grid, hemisphere, benchmarks, model_outputs):
    mean_output = np.array([np.mean(np.abs(getattr(e, grid))) for e in model_outputs[hemisphere]])
    sum_output = np.array([np.sum(np.abs(getattr(e, grid))) for e in model_outputs[hemisphere]])

    assert mean_output == pytest.approx(benchmarks[hemisphere][f"{grid}_mean"].values)
    assert sum_output == pytest.approx(benchmarks[hemisphere][f"{grid}_sum"].values)
