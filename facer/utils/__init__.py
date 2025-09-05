# The Field-Aligned Currents Estimated from Reconnection (FACER) model.
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
from pathlib import Path

def save_maximum_plasma_production_values(lompe_data, target_path):
    """
    Read in data from Lompe at the specified path and save it as a CSV with the accompanying solar zenith angles.
    The relevant file should be in the Lompe repository at lompe/data/chapman_euv_productionvalues.txt.
    (This function created the file in this repository at facer/data/maximum_plasma_production.csv.)

    Parameters
    ----------
    lompe_data : Path
    target_path : Path
    """
    target_file = Path(target_path) / "maximum_plasma_production.csv"

    chi = np.arange(0, 120.1, 0.1)
    product = np.loadtxt(lompe_data)
    array_to_write = np.array([chi, product]).T

    np.savetxt(target_file,
               array_to_write,
               fmt=["%.1f", "%f"],
               delimiter=",",
               header="# Solutions to Equation 26 of Laundal et al. (2022, https://doi.org/10.1029/2022JA030356)\n"
                      "# which gives maximum plasma production q' as a function of solar zenith angle chi.\n"
                      "# Solutions from Lompe (Laundal et al., https://doi.org/10.5281/zenodo.5973739), assuming:\n"
                      "# n(z)        =  1e13 m^-3\n"
                      "# z0          =   500 km\n"
                      "# H           =    50 km\n"
                      "# tau(z, chi) = 1e-20 m^-2\n"
                      "chi,q_dash",
               comments="")