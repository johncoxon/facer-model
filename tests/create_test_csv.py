# Create a csv file which can be used for unit testing the Model class.
import numpy as np
import types
from birkeland import Model
from importlib.machinery import SourceFileLoader
from pandas import DataFrame
from pathlib import Path

loader = SourceFileLoader("birkeland_tests", "Repositories/birkeland/tests/test.py")
birkeland_tests = types.ModuleType(loader.name)
loader.exec_module(birkeland_tests)

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

test_dictionary, _, _ = birkeland_tests.create_test_set(examples)
test_data.update(test_dictionary)

test_data = DataFrame(test_data)
test_data.to_csv(Path(__file__).parent / "test_data.csv")
