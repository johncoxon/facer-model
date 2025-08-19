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
from base_model import BaseModel
from pandas import read_csv
from pathlib import Path


class Model(BaseModel):
    def __init__(self, phi_d, phi_n, f_107, time, hemisphere, sigma_h=12, sigma_p=7,
                 precipitation_conductance="add", flat_earth=False, **kwargs):
        """
        A Python implementation of the Birkeland current model presented by Coxon et al. (2016).

        An expansion of the Milan (2013) model expanded with more realistic conductances based on the
        Moen and Brekke (1993) model of quiet-time conductance.

        kwargs are passed onto the underlying BaseModel class.

        Parameters
        ----------
        phi_d, phi_n : float
            Dayside and nightside reconnection rates, in kV.
        f_107 : float
            The F10.7 index, in solar flux units.
        time : datetime.datetime
        hemisphere : basestring
        sigma_h, sigma_p : float, optional
            The values of the precipitation-driven Hall and Pedersen conductance.
        precipitation_conductance : basestring, optional, default "add"
            Can take one of three options, changing how the model reconciles the return flow region
            between the precipitation-driven conductances and the quiet-time conductances.

            add : Add the precipitation-driven conductances and quiet-time conductances.
            max : Take the maximum of either precipitation-driven or quiet-time conductances
                  (this is the original IDL behaviour).
            replace : Replace the quiet-time with the precipitation-driven conductances.
        flat_earth : bool, optional, default False
            If False, quiet-time conductance modelling uses the modified version of Moen and Brekke (1993) presented by
                Laundal et al. (2022), which modifies the assumption Moen and Brekke (1993) employ of a flat Earth.
            If True, quiet-time conductance modelling uses Moen and Brekke (1993) directly.
        """
        BaseModel.__init__(self, phi_d, phi_n, **kwargs)

        for arg in (f_107, sigma_h, sigma_p):
            if arg:
                if np.isnan(arg):
                    raise ValueError("NaN detected in input.")

        self.f_107 = f_107
        self.time = time

        if hemisphere not in {"north", "south"}:
            raise ValueError("hemisphere must be \"north\" or \"south\".")
        else:
            self.hemisphere = hemisphere

        if precipitation_conductance not in {"add", "max", "replace"}:
            raise ValueError("precipitation_conductance must be \"add\", \"max\", or \"replace\".")
        else:
            self.precipitation_conductance = precipitation_conductance

        self.flat_earth = flat_earth
        self.chi = self.chi_grid()
        self.sigma_h, self.sigma_p = self.sigma_grid(sigma_h, sigma_p)
        self.div_jp, self.div_jh = self.div_j_grid()
        self.j = self.div_jp + self.div_jh

    def chi_grid(self):
        """Grid of solar zenith angle from Ecological Climatology (Bonan, 2015, p. 61)."""
        labda_grid = np.broadcast_to(self.labda, (self._n_theta, self._n_labda)).T
        theta_grid = np.broadcast_to(self.theta, (self._n_labda, self._n_theta))

        doy = self.time.timetuple().tm_yday
        ut = self.time.hour + (self.time.minute / 60.0) + (self.time.second / 3600.0)

        solstice = {"north": 172, "south": 356}
        noon = {"north": 17, "south": 5}
        h = self.hemisphere

        declination = np.radians(23.5 * np.cos(2 * np.pi * (doy - solstice[h]) / 365.25)
                                 + 10 * np.cos(2 * np.pi * (ut - noon[h]) / 24))

        # See p. 6 of lab book for the derivation of this form of Z from Bonan.
        z = np.arccos((np.cos(labda_grid) * np.sin(declination))
                      - (np.sin(labda_grid) * np.cos(declination) * np.cos(theta_grid)))

        return z

    def sigma_q_grid(self):
        """
        Grids of quiet-time Hall and Pedersen conductance, using either Moen and Brekke (1993) or the modified version
        of their method presented by Laundal et al. (2022) depending on the value of self.flat_earth.
        """
        # Follow Moen and Brekke (1993). Equation numbers are from that paper.
        if self.flat_earth:
            # Equation 6 is only real for solar zenith angles less than 90°, so correct any angles greater than that.
            invalid_mask = self.chi > np.radians(90)
            valid_chi = self.chi.copy()
            valid_chi[invalid_mask] = np.radians(90)

            chi_function = np.cos(valid_chi)  # This is the chi function for Moen and Brekke (1993)'s Equation 6.

        # Or follow Laundal et al. (2022). Equation numbers are from that paper.
        else:
            # Read in the maximum plasma production values.
            parent_directory = Path(__file__).parent.resolve()
            production = read_csv(parent_directory / "data" / "maximum_plasma_production.csv", comment="#")

            # Equation 26 is only calculated for solar zenith angles between 0–120°. We can assume any solar zenith
            # angle greater than 120° has a production of zero given that q' is flat for angles greater than ~113°.
            invalid_mask = self.chi > np.radians(120)
            valid_chi = self.chi.copy()
            valid_chi[invalid_mask] = np.radians(120)

            # Get the values of q' for the given solar zenith angles, which is the replacement chi function used in
            # Equations 27 and 28.
            indices = np.searchsorted(np.radians(production.chi), valid_chi)
            chi_function = np.take(production.q_dash.values, indices)

        # Calculate the ionospheric conductances using the functional form from Moen and Brekke (1993), Equation 6.
        sigma_h = (self.f_107 ** 0.53) * ((0.81 * chi_function) + (0.54 * np.sqrt(chi_function)))
        sigma_p = (self.f_107 ** 0.49) * ((0.34 * chi_function) + (0.93 * np.sqrt(chi_function)))

        return sigma_h, sigma_p

    def sigma_grid(self, rf_sigma_h, rf_sigma_p):
        """
        Combine the quiet-time grid from Moen and Brekke (1993) with the user-specified Hall and
        Pedersen return-flow-region conductivities.

        Parameters
        ----------
        rf_sigma_h, rf_sigma_p : float
            The values of the Hall and Pedersen conductivities in the return flow region.

        Returns
        -------
        sigma_h, sigma_p : np.ndarray
            Arrays of the Hall and Pedersen conductivity on the model grid.
        """
        sigma_h, sigma_p = self.sigma_q_grid()

        # Set the Pedersen and Hall conductivities in the return flow region.
        _, _, _, mask = self.labda_by_region()

        if self.precipitation_conductance == "add":
            sigma_h[mask, :] += rf_sigma_h
            sigma_p[mask, :] += rf_sigma_p
        elif self.precipitation_conductance == "max":
            mask_grid = np.broadcast_to(mask, (self._n_theta, self._n_labda)).T

            mask_h = mask_grid & (sigma_h < rf_sigma_h)
            sigma_h[mask_h] = rf_sigma_h

            mask_p = mask_grid & (sigma_p < rf_sigma_p)
            sigma_p[mask_p] = rf_sigma_p
        else:
            sigma_h[mask, :] = rf_sigma_h
            sigma_p[mask, :] = rf_sigma_p

        return sigma_h, sigma_p

    def div_j_grid(self):
        """Grids of the divergence of Hall and Pedersen current (calculated from E and sigma)."""
        div_jp = np.zeros_like(self.sigma_p)
        div_jh = np.zeros_like(self.sigma_h)

        sin_colat = np.sin(np.radians(self.colat[:-1] + 0.5))
        j_plus_1 = np.concatenate((np.arange(359) + 1, [0]))

        l_labda = (2 * np.pi * self._r_e / self._n_theta)
        l_theta = np.expand_dims((2 * np.pi * self._r_e * sin_colat / self._n_theta), axis=1)

        j_p_labda = l_theta * (self.e_labda[:-1, :] * self.sigma_p[:-1, :]
                               - self.e_labda[1:, :] * self.sigma_p[1:, :])
        j_p_theta = -l_labda * (self.e_theta[1:, j_plus_1] * self.sigma_p[1:, j_plus_1]
                                - self.e_theta[1:, :] * self.sigma_p[1:, :])

        j_h_labda = l_theta * (self.e_theta[:-1, :] * self.sigma_h[:-1, :]
                               - self.e_theta[1:, :] * self.sigma_h[1:, :])
        j_h_theta = l_labda * (self.e_labda[1:, j_plus_1] * self.sigma_h[1:, j_plus_1]
                               - self.e_labda[1:, :] * self.sigma_h[1:, :])

        div_jp[1:, :] = j_p_labda + j_p_theta
        div_jh[1:, :] = j_h_labda + j_h_theta

        return div_jp, div_jh

    def map_solar_zenith_angle(self, ax, vmin=45, vmax=135, cmap="magma_r", contours=True,
                               **kwargs):
        """Plot a map of the solar zenith angle."""
        mesh = self._plot_map(ax, np.degrees(self.chi), vmin, vmax, cmap, contours, **kwargs)
        return mesh

    def map_sigma(self, ax, component, vmin=0, vmax=10, cmap="viridis", contours=True, **kwargs):
        if component.lower() == "hall":
            sigma = self.sigma_h
        elif component.lower() == "pedersen":
            sigma = self.sigma_p
        else:
            raise ValueError("Component must be \"hall\" or \"pedersen\".")

        mesh = self._plot_map(ax, sigma, vmin, vmax, cmap, contours, **kwargs)
        self._annotate_map(ax, component)

        return mesh

    def map_div_j(self, ax, component, vlim=6, cmap="RdBu_r", contours=True, **kwargs):
        if component.lower() == "hall":
            div_j = self.div_jh / 1e3
        elif component.lower() == "pedersen":
            div_j = self.div_jp / 1e3
        else:
            raise ValueError("Component must be \"hall\" or \"pedersen\".")

        mesh = self._plot_map(ax, div_j, -vlim, vlim, cmap, contours, **kwargs)
        self._annotate_map(ax, component)

        return mesh

    @staticmethod
    def _annotate_map(ax, component):
        if component == "labda":
            annotation = r"$\mathregular{\lambda}$"
        elif component == "theta":
            annotation = r"$\mathregular{\theta}$"
        elif component.lower() == "hall":
            annotation = "H"
        else:
            annotation = "P"

        ax.annotate(annotation, xy=(1, 1), xycoords="axes fraction",
                    xytext=(-5, -5), textcoords="offset points",
                    fontsize="xx-large", ha="right", va="top")