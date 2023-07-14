import numpy as np
from birkeland import Model
from pandas import read_csv
from pathlib import Path
from unittest import TestCase


class TestBirkeland(TestCase):
    def test_against_data(self):
        benchmarks = read_csv(Path(__file__).parent / "test_data.csv")
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


def create_test_set(examples):
    """
    Create a dictionary which can either be saved to disk to act as benchmarks for future tests, or
    can be used to compare against the benchmarks upon installation.

    Parameters
    ----------
    examples : np.ndarray
        An array of Model.

    Returns
    -------
    test_dictionary : dict
        A dictionary containing keys and the outputs for each key from the input array of Model.
    """
    test_dictionary = {}

    labda_values = np.arange(0, 35, 5, dtype=int)
    for labda in labda_values:
        b_r = np.array([e.b_r(labda) for e in examples])
        test_dictionary[f"b_r_{labda}"] = b_r

    methods = ("lambda_r1", "v_r1", "e_b", "e_d", "e_n")
    for key in methods:
        test_dictionary[key] = np.array([getattr(e, key)() for e in examples])

    grids = ("phi_r1", "phi_grid", "e_grid", "v_grid", "j_grid")
    for key in grids:
        test_dictionary[f"{key}_mean"] = np.array([np.mean(np.abs(getattr(e, key)()))
                                                   for e in examples])
        test_dictionary[f"{key}_sum"] = np.array([np.sum(np.abs(getattr(e, key)()))
                                                  for e in examples])

    return test_dictionary, methods, grids
