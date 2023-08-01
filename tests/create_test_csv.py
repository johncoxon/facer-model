# Create a csv file which can be used for unit testing the Model class.
import numpy as np
import types
from birkeland import BetterModel, Model
from datetime import datetime
from importlib.machinery import SourceFileLoader
from pandas import DataFrame
from pathlib import Path

loader = SourceFileLoader("birkeland_tests", "Repositories/birkeland/tests/test.py")
birkeland_tests = types.ModuleType(loader.name)
loader.exec_module(birkeland_tests)

types = ["model", "north", "south"]

examples = {}
for t in types:
    examples[t] = np.empty((6, 6, 5), dtype=object)

phi_d_values = np.arange(0, 60, 10)
phi_n_values = np.arange(0, 60, 10)
f_pc_values = np.arange(0.1, 1.1, 0.2)

test_data = {"phi_d": [],
             "phi_n": [],
             "f_pc": []}

f_107 = 100

for cnt_d, phi_d in enumerate(phi_d_values):
    for cnt_n, phi_n in enumerate(phi_n_values):
        for cnt_f, f_pc in enumerate(f_pc_values):
            examples["model"][cnt_d, cnt_n, cnt_f] = Model(phi_d, phi_n, f_pc=f_pc)

            for t in ["north", "south"]:
                examples[t][cnt_d, cnt_n, cnt_f] = BetterModel(phi_d, phi_n, f_107,
                                                               datetime(2010, 1, 1), t, f_pc=f_pc)

            test_data["phi_d"].append(phi_d)
            test_data["phi_n"].append(phi_n)
            test_data["f_pc"].append(f_pc)

for key in test_data:
    test_data[key] = np.array(test_data[key])

for t in ["model", "north", "south"]:
    test_examples = examples[t].flatten()
    test_data_copy = dict(test_data)

    test_dictionary, _, _ = birkeland_tests.create_test_set(test_examples, model_type=t)
    test_data_copy.update(test_dictionary)

    test_dictionary = DataFrame(test_data_copy)
    test_dictionary.to_csv(Path(__file__).parent / f"test_data_{t}.csv")
