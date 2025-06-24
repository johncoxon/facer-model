# Tests for the FACER model.
# Copyright (C) 2025 John Coxon (work@johncoxon.co.uk)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pytest
from datetime import datetime
from facer import BaseModel, Model
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
    model_outputs = []

    for cnt, row in benchmarks["model"].iterrows():
        model_outputs.append(BaseModel(row["phi_d"], row["phi_n"], row["f_pc"]))

    model_outputs = np.array(model_outputs)

    return model_outputs


@pytest.fixture
def better_model_outputs(benchmarks):
    better_model_outputs = {}
    f_107 = 100

    for h in ("north", "south"):
        better_model_outputs[h] = []

        for cnt, row in benchmarks["model"].iterrows():
            better_model_outputs[h].append(Model(row["phi_d"], row["phi_n"], f_107,
                                           datetime(2010, 1, 1, 17), h, f_pc=row["f_pc"]))
        better_model_outputs[h] = np.array(better_model_outputs[h])

    return better_model_outputs


@pytest.mark.parametrize("b_r", (0, 5, 10, 15, 20, 25, 30))
def test_b_r(b_r, benchmarks, model_outputs):
    output = np.array([e.b_r(b_r) for e in model_outputs])
    assert output == pytest.approx(benchmarks["model"][f"b_r_{b_r}"].values)


@pytest.mark.parametrize("method", ("lambda_r1", "v_r1", "e_b", "e_d", "e_n"))
def test_methods(method, benchmarks, model_outputs):
    output = np.array([getattr(e, method)() for e in model_outputs])
    assert output == pytest.approx(benchmarks["model"][method].values)


@pytest.mark.parametrize("grid", ("phi", "e_labda", "e_theta", "v_labda", "v_theta", "j"))
def test_model(grid, benchmarks, model_outputs):
    median_output = np.array([np.median(np.abs(getattr(e, grid))) for e in model_outputs])
    sum_output = np.array([np.sum(np.abs(getattr(e, grid))) for e in model_outputs])

    assert median_output == pytest.approx(benchmarks["model"][f"{grid}_median"].values)
    assert sum_output == pytest.approx(benchmarks["model"][f"{grid}_sum"].values)


@pytest.mark.parametrize("grid", ("sza", "sigma_h", "sigma_p", "div_jh", "div_jp"))
@pytest.mark.parametrize("hemisphere", ("north", "south"))
def test_better_model(grid, hemisphere, benchmarks, better_model_outputs):
    median_output = np.array([np.median(np.abs(getattr(e, grid)))
                            for e in better_model_outputs[hemisphere]])
    sum_output = np.array([np.sum(np.abs(getattr(e, grid)))
                           for e in better_model_outputs[hemisphere]])

    assert median_output == pytest.approx(benchmarks[hemisphere][f"{grid}_median"].values)
    assert sum_output == pytest.approx(benchmarks[hemisphere][f"{grid}_sum"].values)
