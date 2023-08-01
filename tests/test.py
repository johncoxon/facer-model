import numpy as np
from birkeland import Model, BetterModel
from datetime import datetime
from pandas import read_csv
from pathlib import Path
from unittest import TestCase


class TestBirkeland(TestCase):
    def test_model(self, model_type="model"):
        if model_type == "model":
            error_prefix = f"Model, "
        else:
            error_prefix = f"BetterModel, h={model_type}, "

        filename = f"test_data_{model_type}.csv"

        benchmarks = read_csv(Path(__file__).parent / filename)
        examples = []

        f_107 = 100

        for cnt, row in benchmarks.iterrows():
            if model_type == "model":
                examples.append(Model(row["phi_d"], row["phi_n"], row["f_pc"]))
            else:
                examples.append(BetterModel(row["phi_d"], row["phi_n"], f_107,
                                            datetime(2010, 1, 1), model_type, f_pc=row["f_pc"]))

        examples = np.array(examples)

        data, methods, grids = create_test_set(examples, model_type=model_type)

        for key in methods:
            np.testing.assert_allclose(data[key], benchmarks[key].values,
                                       err_msg=error_prefix + key)

        for key in grids:
            np.testing.assert_allclose(data[f"{key}_mean"], benchmarks[f"{key}_mean"].values,
                                       err_msg=error_prefix + f"{key} mean")
            np.testing.assert_allclose(data[f"{key}_sum"], benchmarks[f"{key}_sum"].values,
                                       err_msg=error_prefix + f"{key} sum")

    def test_north(self):
        self.test_model(model_type="north")

    def test_south(self):
        self.test_model(model_type="south")


def create_test_set(examples, model_type="model"):
    """
    Create a dictionary which can either be saved to disk to act as benchmarks for future tests, or
    can be used to compare against the benchmarks upon installation.

    Parameters
    ----------
    examples : np.ndarray
        An array of Model or BetterModel.
    model_type : basestring, optional, default "model"
        Create tests for BetterModel.

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

    if model_type == "model":
        grids = ("phi", "e_labda", "e_theta", "v_labda", "v_theta", "j")
    else:
        grids = ("sza", "sigma_h", "sigma_p", "div_jh", "div_jp")

    for key in grids:
        test_dictionary[f"{key}_mean"] = np.array([np.mean(np.abs(getattr(e, key)))
                                                   for e in examples])
        test_dictionary[f"{key}_sum"] = np.array([np.sum(np.abs(getattr(e, key)))
                                                  for e in examples])

    return test_dictionary, methods, grids
