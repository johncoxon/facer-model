import numpy as np
from birkeland import Model
from pandas import DataFrame, read_csv
from pathlib import Path
from unittest import TestCase

test_directory = Path(__file__).parent


class TestBirkeland(TestCase):
    def test_against_data(self):
        benchmarks = read_csv(test_directory / "test_data.csv")
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


def create_test_csv(filepath):
    """
    Create a csv file which can be used for unit testing the Model class.

    Parameters
    ----------
    filepath : pathlib.Path
        The filepath of the csv file to act as benchmarks for future unit testing.
    """
    examples = np.empty((6, 6, 5), dtype=object)
    phi_d_values = np.arange(0, 60, 10)
    phi_n_values = np.arange(0, 60, 10)
    f_pc_values = np.arange(0.1, 1.1, 0.2)

    test_data = {"phi_d": [],
                 "phi_n": [],
                 "f_pc": []}

    for cnt_d, phi_d in enumerate(phi_d_values):
        for cnt_n, phi_n in enumerate(phi_n_values):
            for cnt_f, f_pc in enumerate(f_pc_values):
                examples[cnt_d, cnt_n, cnt_f] = Model(phi_d, phi_n, f_pc)

                test_data["phi_d"].append(phi_d)
                test_data["phi_n"].append(phi_n)
                test_data["f_pc"].append(f_pc)

    examples = examples.flatten()

    for key in test_data:
        test_data[key] = np.array(test_data[key])

    test_dictionary, _, _ = create_test_set(examples)
    test_data.update(test_dictionary)

    test_data = DataFrame(test_data)
    test_data.to_csv(filepath)


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
