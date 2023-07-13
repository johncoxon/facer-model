import numpy as np
from birkeland import Model, create_test_set
from pandas import read_csv
from pathlib import Path
from unittest import TestCase


class TestBirkeland(TestCase):
    def test_against_data(self):
        benchmarks = read_csv(Path("~/Repositories/Python/ampere/tests/test_data.csv").expanduser())
        examples = []
        for cnt, row in benchmarks.iterrows():
            examples.append(Model(row["phi_d"], row["phi_n"], row["f_pc"]))
        examples = np.array(examples)

        data, methods, grids = create_test_set(examples)

        for key in methods:
            np.testing.assert_allclose(data[key], benchmarks[key].values, err_msg=key)

        for key in grids:
            np.testing.assert_allclose(data[f"{key}_mean"], benchmarks[f"{key}_mean"].values,
                                       err_msg=f"{key} mean")
            np.testing.assert_allclose(data[f"{key}_sum"], benchmarks[f"{key}_sum"].values,
                                       err_msg=f"{key} sum")
