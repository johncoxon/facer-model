# Create a csv file which can be used for unit testing the BaseModel class.
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
from datetime import datetime
from facer import BaseModel, Model
from pandas import DataFrame
from pathlib import Path


def create_test_set(examples, model_type="model"):
    """
    Create a dictionary which can either be saved to disk to act as benchmarks for future tests, or
    can be used to compare against the benchmarks upon installation.

    Parameters
    ----------
    examples : np.ndarray
        An array of BaseModel or Model.
    model_type : basestring, optional, default "model"
        If "model", create tests for BaseModel.
        If "north" or "south", create tests for Model with that hemisphere.

    Returns
    -------
    test_dictionary : dict
        A dictionary containing keys and the outputs for each key from the input array.
    """
    dictionary = {}

    labda_values = np.arange(0, 35, 5, dtype=int)
    for labda in labda_values:
        b_r = np.array([e.b_r(labda) for e in examples])
        dictionary[f"b_r_{labda}"] = b_r

    methods = ("lambda_r1", "v_r1", "e_b", "e_d", "e_n")
    for method in methods:
        dictionary[method] = np.array([getattr(e, method)() for e in examples])

    if model_type == "model":
        grids = ("phi", "e_labda", "e_theta", "v_labda", "v_theta", "j")
    else:
        grids = ("sza", "sigma_h", "sigma_p", "div_jh", "div_jp")

    for grid in grids:
        dictionary[f"{grid}_median"] = np.array([np.median(np.abs(getattr(e, grid)))
                                               for e in examples])
        dictionary[f"{grid}_sum"] = np.array([np.sum(np.abs(getattr(e, grid)))
                                              for e in examples])

    return dictionary, methods, grids


types = ["model", "north", "south"]

model_outputs = {}
for t in types:
    model_outputs[t] = np.empty((6, 6, 5), dtype=object)

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
            model_outputs["model"][cnt_d, cnt_n, cnt_f] = BaseModel(phi_d, phi_n, f_pc=f_pc)

            for t in ["north", "south"]:
                model_outputs[t][cnt_d, cnt_n, cnt_f] = Model(phi_d, phi_n, f_107,
                                                              datetime(2010, 1, 1, 17),
                                                              t, f_pc=f_pc)

            test_data["phi_d"].append(phi_d)
            test_data["phi_n"].append(phi_n)
            test_data["f_pc"].append(f_pc)

for key in test_data:
    test_data[key] = np.array(test_data[key])

for t in ["model", "north", "south"]:
    test_model_outputs = model_outputs[t].flatten()
    test_data_copy = dict(test_data)

    test_dictionary, _, _ = create_test_set(test_model_outputs, model_type=t)
    test_data_copy.update(test_dictionary)

    test_dictionary = DataFrame(test_data_copy)
    test_dictionary.to_csv(Path(__file__).parent / f"test_data_{t}.csv")
