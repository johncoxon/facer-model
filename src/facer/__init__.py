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
import warnings
from datetime import timedelta
from matplotlib.ticker import FuncFormatter, MultipleLocator


class BaseModel(object):
    def __init__(self, phi_d, phi_n, f_pc=None, r1_colat=None, delta_colat=10,
                 theta_d=30, theta_n=30, sigma_pc=1, sigma_rf=1, order_n=20):
        """
        A Python implementation of the Birkeland current model presented by Milan (2013).

        A simple mathematical model of the region 1 and 2 Birkeland current system intensities for
        differing dayside and nightside magnetic reconnection rates, consistent with the
        expanding/contracting polar cap paradigm of solar wind-magnetosphere-ionosphere coupling.

        Parameters
        ----------
        phi_d, phi_n : float
            Dayside and nightside reconnection rates, in kV.
        f_pc : float, optional, default None
            Polar cap flux in GWb. If set, labda_r1 is calculated from this value.
        r1_colat : float, optional, default None
            Colatitude of the R1 current oval in deg. If set, f_pc is calculated from this value.
        delta_colat : float, optional, default 10
            The colatitudinal gap between the R1 and R2 current ovals in degrees.
        theta_d, theta_n : float, optional, default 30
            The azimuthal widths of the dayside and nightside merging gaps, in degrees.
        sigma_pc, sigma_rf : float, optional, default 1
            The conductivities in polar cap and return flow regions, in mho.
        order_n : int, optional, default 20
            Set this to govern the order of the Fourier terms in the model.
        """
        for arg in (phi_d, phi_n, f_pc, r1_colat, delta_colat, theta_d, theta_n, sigma_pc, sigma_rf, order_n):
            if arg:
                if np.isnan(arg):
                    raise ValueError("NaN detected in input.")

        self.phi_d = phi_d * 1e3
        self.phi_n = phi_n * 1e3                        # Convert to SI units from inputs
        self.theta_d = np.radians(theta_d)
        self.theta_n = np.radians(theta_n)
        self.sigma_pc = sigma_pc
        self.sigma_rf = sigma_rf

        # Define the ratio of conductivities (it makes the maths for R1 current easier).
        self._alpha = self.sigma_rf / self.sigma_pc

        # Configure magnetic field information for the model.
        self._r_e = 6.371e6                     # Earth radius of 6371 km.
        self._b_eq = 31000e-9                   # Equatorial field strength of 31,000 nT.

        if (f_pc is not None) and (r1_colat is not None):
            self.f_pc = f_pc * 1e9
            self.labda_r1 = np.radians(r1_colat)
            warnings.warn("Setting both polar cap flux and R1 colatitude will set both manually. "
                          "This is supported to allow comparisons with the original IDL, but is "
                          "not recommended.")
        elif f_pc is not None:
            self.f_pc = f_pc * 1e9
            self.labda_r1 = self.lambda_r1()
        elif r1_colat is not None:
            self.labda_r1 = np.radians(r1_colat)
            self.f_pc = self.f_pc_analytic()
        else:
            raise ValueError("You must pass either polar cap flux or R1 colatitude to the model.")
        self.labda_r2 = self.labda_r1 + np.radians(delta_colat)

        # Configure the default grid for the model. Milan (2013) uses the symbol lambda to refer to
        # colatitude, but lambda has an inbuilt meaning in Python, so we use "labda" instead.
        self._n_labda = 31
        self._n_theta = 360
        self.labda = np.radians(np.linspace(0.5, 30.5, self._n_labda))   # 1° latitudinal resolution
        self.theta = np.radians(np.linspace(0.5, 359.5, self._n_theta))  # 1° azimuthal resolution
        self.colat = np.degrees(self.labda)
        self.mlt = np.degrees(self.theta) / 15

        # Set the limit of the Fourier series used in the maths and calculate the s_m variable.
        # Add one to order_n so that range returns m from 1 up to order_n.
        self._m = np.expand_dims(np.arange(1, order_n + 1), axis=0)
        self._s_m = self.s_m_analytic()

        # Obtain the key variables on the grid of lambda and theta defined above.
        self.phi = self.phi_grid()
        self.e_labda, self.e_theta = self.e_grid()
        self.v_labda, self.v_theta = self.v_grid()
        self.j = self.j_grid()

    def b_r(self, labda):
        """Radial magnetic field from Equation 6."""
        return 2 * self._b_eq * np.cos(labda)

    def b_r_grid(self):
        """Radial magnetic field on a grid."""
        b_r = np.broadcast_to(self.b_r(self.labda), (self._n_theta, self._n_labda)).T
        return b_r

    def f_pc_analytic(self):
        """F_PC determined using lambda_R1 from Equation 8."""
        return 2 * np.pi * (self._r_e ** 2) * self._b_eq * (np.sin(self.labda_r1) ** 2)

    def lambda_r1(self):
        """lambda_R1 determined using F_PC from the inverse of Equation 8."""
        return np.arcsin(np.sqrt(self.f_pc / (2 * np.pi * (self._r_e ** 2) * self._b_eq)))

    def v_r1(self):
        """
        R1 current oval velocity V_R1 from Equation 9. Note that we do not take the square of Earth
        radius; this appears to be a typo in the paper, as in the IDL code the equation appears as
        expressed here.
        """
        numerator = self.phi_d - self.phi_n
        denominator = 2 * np.pi * self._r_e * self._b_eq * np.sin(2 * self.labda_r1)
        return numerator / denominator

    def e_b(self):
        """Electric field at nonreconnecting regions of the boundary from Equation 11."""
        return -self.v_r1() * self.b_r(self.labda_r1)

    def e_d(self):
        """Electric field in the dayside merging gap from Equation 12."""
        l_d = 2 * self.theta_d * self._r_e * np.sin(self.labda_r1)
        return self.e_b() + self.phi_d / l_d

    def e_n(self):
        """Electric field in the nightside merging gap from Equation 13."""
        l_n = 2 * self.theta_n * self._r_e * np.sin(self.labda_r1)
        return self.e_b() - self.phi_n / l_n

    def phi_r1(self):
        """R1 current system electric potential as a function of theta from Table 1."""
        phi_r1 = np.ones_like(self.theta) * np.nan

        # These are the six different boundary conditions in Table 1.
        condition0 = 0
        condition1 = self.theta_n
        condition2 = np.pi - self.theta_d
        condition3 = np.pi + self.theta_d
        condition4 = 2 * np.pi - self.theta_n
        condition5 = 2 * np.pi

        # These are the five different combinations of conditions in Table 1.
        mask1 = ((self.theta >= condition0) & (self.theta < condition1))
        mask2 = ((self.theta >= condition1) & (self.theta < condition2))
        mask3 = ((self.theta >= condition2) & (self.theta < condition3))
        mask4 = ((self.theta >= condition3) & (self.theta < condition4))
        mask5 = ((self.theta >= condition4) & (self.theta < condition5))

        # This actually does the maths from Table 1 and puts it in the array. There isn't a
        # great way to make this human-readable, unfortunately.
        phi_r1[mask1] = self.e_n() * self.theta[mask1]
        phi_r1[mask2] = ((self.e_n() - self.e_b()) * self.theta_n
                         + self.e_b() * self.theta[mask2])
        phi_r1[mask3] = ((self.e_n() - self.e_b()) * self.theta_n
                         + (self.e_d() - self.e_b()) * (self.theta_d - np.pi)
                         + self.e_d() * self.theta[mask3])
        phi_r1[mask4] = ((self.e_n() - self.e_b()) * self.theta_n
                         + 2 * (self.e_d() - self.e_b()) * self.theta_d
                         + self.e_b() * self.theta[mask4])
        phi_r1[mask5] = (2 * (self.e_n() - self.e_b()) * (self.theta_n - np.pi)
                         + 2 * (self.e_d() - self.e_b()) * self.theta_d
                         + self.e_n() * self.theta[mask5])

        # Every solution from Table 1 is multiplied by -R_E * np.sin(labda_R1), so do that now.
        phi_r1 *= -self._r_e * np.sin(self.labda_r1)

        if phi_r1.shape[0] == 1:
            phi_r1 = phi_r1[0]

        return phi_r1

    @staticmethod
    def _lambda(labda):
        """The capital lambda term defined on page 5 and used in Equations 17-19."""
        return np.log(np.tan(labda / 2))

    def phi_pc(self, labda):
        """Polar cap potential calculated from Equation 17."""
        theta = np.expand_dims(self.theta, axis=1)

        sine_term = self._s_m * np.sin(theta @ self._m)
        exp_term = np.exp(self._m.T @ (self._lambda(labda) - self._lambda(self.labda_r1)))
        phi_pc = (sine_term @ exp_term).T

        return phi_pc

    def phi_rf(self, labda):
        """Return flow potential calculated from Equation 18."""
        theta = np.expand_dims(self.theta, axis=1)

        sine_term = self._s_m * np.sin(theta @ self._m)
        numerator = np.sinh(self._m.T @ (self._lambda(labda) - self._lambda(self.labda_r2)))
        denominator = np.sinh(
            self._m.T * (self._lambda(self.labda_r1) - self._lambda(self.labda_r2)))

        phi_rf = (sine_term @ (numerator / denominator)).T

        return phi_rf

    def labda_by_region(self):
        """
        The labda for the polar cap (poleward of R1 colatitude) and return flow (between R1 and R2
        colatitudes) regions. Doesn't return low-latitude (equatorward of R2) region since
        everything is zero at those colatitudes.
        """
        polar_cap_mask = (self.labda < self.labda_r1)
        return_flow_mask = ((self.labda >= self.labda_r1) & (self.labda < self.labda_r2))

        polar_cap_labda = np.expand_dims(self.labda[polar_cap_mask], axis=0)
        return_flow_labda = np.expand_dims(self.labda[return_flow_mask], axis=0)

        return polar_cap_labda, polar_cap_mask, return_flow_labda, return_flow_mask

    def phi_grid(self):
        """
        The electric potential from Equations 17-19 depending on the region in which colatitude lies
        (polar cap, return flow, or low-latitude region) on the underlying model grid.
        """
        phi = np.zeros((self.labda.shape[0], self.theta.shape[0]))

        labda_pc, mask_pc, labda_rf, mask_rf = self.labda_by_region()
        phi[mask_pc, :] = self.phi_pc(labda_pc)
        phi[mask_rf, :] = self.phi_rf(labda_rf)

        return phi

    def s_m_analytic(self):
        """Fourier expansion of phi_R1 analytically from Equation 21."""
        d_term = (self.phi_d * np.sin(self._m * self.theta_d) / self.theta_d) * ((-1) ** self._m)
        n_term = (self.phi_n * np.sin(self._m * self.theta_n) / self.theta_n)

        s_m = ((-1 / (np.pi * (self._m ** 2))) * (d_term - n_term)).squeeze()

        return s_m

    def partial_differential_of_phi_pc(self, labda, differentiate_by):
        """Partial differentials of Phi_PC, from Equations 22 and 23."""
        theta = np.expand_dims(self.theta, axis=1)

        if differentiate_by == "labda":
            trig_term = np.sin(theta @ self._m)
        elif differentiate_by == "theta":
            trig_term = np.cos(theta @ self._m)
        else:
            raise ValueError("Value of 'differentiate_by' must be 'labda' or 'theta'.")
        exp_term = np.exp(self._m.T @ (self._lambda(labda) - self._lambda(self.labda_r1)))

        diff = ((self._s_m * self._m * trig_term) @ exp_term).T

        return diff

    def partial_differential_of_phi_rf(self, labda, differentiate_by):
        """Partial differentials of Phi_RF, from Equation 24 and 25."""
        theta = np.expand_dims(self.theta, axis=1)

        if differentiate_by == "labda":
            trig_term = np.sin(theta @ self._m)
            numerator = np.cosh(self._m.T @ (self._lambda(labda) - self._lambda(self.labda_r2)))
        elif differentiate_by == "theta":
            trig_term = np.cos(theta @ self._m)
            numerator = np.sinh(self._m.T @ (self._lambda(labda) - self._lambda(self.labda_r2)))
        else:
            raise ValueError("Value of 'differentiate_by' must be 'labda' or 'theta'.")

        denominator = np.sinh(
            self._m.T * (self._lambda(self.labda_r1) - self._lambda(self.labda_r2)))

        diff = ((self._s_m * self._m * trig_term) @ (numerator / denominator)).T

        return diff

    def e(self, labda, component):
        """Either component of electric field from Equation 26."""
        if not isinstance(labda, float):
            raise ValueError("Passing multiple colatitudes doesn't work yet.")

        if labda >= self.labda_r2:              # Equatorward of R2 colatitude => low-latitude
            e = np.zeros_like(self.theta)
        elif labda < self.labda_r1:             # Poleward of R1 colatitude => polar cap
            e = self.partial_differential_of_phi_pc(labda, component)
        else:                                   # Between R1 and R2 colatitudes => return flow
            e = self.partial_differential_of_phi_rf(labda, component)

        e /= -(self._r_e * np.sin(labda))

        return e

    def e_grid(self):
        """Either component of electric field from Equation 26."""
        e_labda = np.zeros((self.labda.shape[0], self.theta.shape[0]))
        e_theta = np.zeros((self.labda.shape[0], self.theta.shape[0]))

        labda_pc, mask_pc, labda_rf, mask_rf = self.labda_by_region()

        e_labda[mask_pc, :] = self.partial_differential_of_phi_pc(labda_pc, "labda")
        e_theta[mask_pc, :] = self.partial_differential_of_phi_pc(labda_pc, "theta")

        e_labda[mask_rf, :] = self.partial_differential_of_phi_rf(labda_rf, "labda")
        e_theta[mask_rf, :] = self.partial_differential_of_phi_rf(labda_rf, "theta")

        divisor = -(self._r_e * np.sin(np.expand_dims(self.labda, axis=1)))
        e_labda /= divisor
        e_theta /= divisor

        return e_labda, e_theta

    def v_grid(self):
        """Either component of the ionospheric flow vector from Equation 27."""
        v_labda = -self.e_theta / self.b_r_grid()
        v_theta = self.e_labda / self.b_r_grid()

        return v_labda, v_theta

    def j_r1_intensity(self, theta=None):
        """The R1 current per unit of azimuthal distance from Equation 28."""
        if not theta:
            theta = np.expand_dims(self.theta, axis=1)

        first_term = self.sigma_pc / (self._r_e * np.sin(self.labda_r1))
        lambda_term = self._lambda(self.labda_r1) - self._lambda(self.labda_r2)

        sin_term = self._s_m * self._m * np.sin(theta @ self._m)
        coth_term = self._alpha * self._coth(self._m * lambda_term) - 1
        integration_term = np.sum(sin_term * coth_term, axis=1)

        return first_term * integration_term

    def j_r2_intensity(self, theta=None):
        """The R2 current per unit of azimuthal distance from Equation 29."""
        if not theta:
            theta = np.expand_dims(self.theta, axis=1)

        first_term = -self.sigma_rf / (self._r_e * np.sin(self.labda_r2))
        lambda_term = self._lambda(self.labda_r1) - self._lambda(self.labda_r2)

        sin_term = self._s_m * self._m * np.sin(theta @ self._m)
        csch_term = self._csch(self._m * lambda_term)
        integration_term = np.sum(sin_term * csch_term, axis=1)

        return first_term * integration_term

    def s_m_odd(self):
        """Get just m and s_m for odd numbers, for Equations 30 and 31."""
        s_m = self._s_m[::2]
        m = self._m.squeeze()[::2]
        return m, s_m

    def j_r1_integrated(self):
        """The R1 current integrated in azimuth from Equation 30."""
        first_term = - 2 * np.pi * self.sigma_pc
        lambda_term = self._lambda(self.labda_r1) - self._lambda(self.labda_r2)
        m, s_m = self.s_m_odd()

        integration_term = np.sum(s_m * (self._alpha * self._coth(m * lambda_term) - 1))

        return first_term * integration_term

    def j_r2_integrated(self):
        """The R2 current integrated in azimuth from Equation 31."""
        first_term = - 2 * np.pi * self.sigma_rf
        lambda_term = self._lambda(self.labda_r1) - self._lambda(self.labda_r2)
        m, s_m = self.s_m_odd()

        integration_term = np.sum(s_m * self._csch(m * lambda_term))

        return first_term * integration_term

    def j_grid(self):
        """
        The field-aligned current on the underlying model grid, assuming that the currents have a
        width of 1° in colatitude and mapping them to the nearest colatitude on the underlying grid.
        """
        j = np.zeros((self.labda.shape[0], self.theta.shape[0]))

        r1_index = np.argmin(np.abs(self.labda - self.labda_r1))
        j[r1_index, :] = self.j_r1_intensity()

        r2_index = np.argmin(np.abs(self.labda - self.labda_r2))
        j[r2_index, :] = self.j_r2_intensity()

        return j

    def j_total(self):
        return np.sum(self.j[self.j > 0]) - np.sum(self.j[self.j < 0])

    @staticmethod
    def _coth(x):
        """Used in Equations 28 and 30."""
        return np.cosh(x) / np.sinh(x)

    @staticmethod
    def _csch(x):
        """Used in Equations 29 and 31."""
        return 1 / np.sinh(x)

    def draw_potential_contours(self, ax):
        """Draw contours of the electric potential as in Figure 2e-j."""
        phi_contours = np.concatenate((np.arange(-95, 0, 10), np.arange(5, 105, 10))) * 1e3
        ax.contour(self.theta, self.colat, self.phi, levels=phi_contours, colors="black")

    def map_current(self, ax, vlim=100, cmap="RdBu_r", contours=True, **kwargs):
        """Plot a map of the Birkeland current as in Figure i-j."""
        mesh = self._plot_map(ax, self.j_grid() * 1e3, -vlim, vlim, cmap, contours, **kwargs)
        return mesh

    def map_electric_field(self, ax, component, vlim=50, cmap="PuOr_r", contours=True, **kwargs):
        """Plot a map of the electric field in either component as in Figure 2e-h."""
        if component == "labda":
            e_field = self.e_labda * 1e3
        elif component == "theta":
            e_field = self.e_theta * 1e3
        else:
            raise ValueError("Component must be \"theta\" or \"phi\".")

        mesh = self._plot_map(ax, e_field, -vlim, vlim, cmap, contours, **kwargs)
        self._annotate_map(ax, component)

        return mesh

    def map_electric_potential(self, ax, vlim=30, cmap="PuOr_r", contours=True, **kwargs):
        """Plot a map of the electric potential."""
        mesh = self._plot_map(ax, self.phi / 1e3, -vlim, vlim, cmap, contours, **kwargs)
        return mesh

    def map_flow_vector(self, ax, component, vlim=750, cmap="PuOr_r", contours=True, **kwargs):
        """Plot a map of the ionospheric flow vector."""
        if component == "labda":
            flow_vector = self.v_labda
        elif component == "theta":
            flow_vector = self.v_theta
        else:
            raise ValueError("Component must be \"theta\" or \"phi\".")

        mesh = self._plot_map(ax, flow_vector, -vlim, vlim, cmap, contours, **kwargs)
        self._annotate_map(ax, component)

        return mesh

    def plot_r1_and_r2_intensity(self, ax):
        """Plot the intensity of Region 1 and Region 2 as in Figure 2c-d."""
        ax.plot(self.mlt, self.j_r1_intensity() * 1e3, label="R1")
        ax.plot(self.mlt, self.j_r2_intensity() * 1e3, label="R2")

        r1_string = fr"$\mathregular{{J_{{R1}}=}}${self.j_r1_integrated() / 1e6:.2f} MA mho$^{-1}$"
        r2_string = fr"$\mathregular{{J_{{R2}}=}}${self.j_r2_integrated() / 1e6:.2f} MA mho$^{-1}$"
        ax.annotate(r1_string + "\n" + r2_string, (1, 0), xycoords="axes fraction", xytext=(-5, 5),
                    textcoords="offset points", ha="right", va="bottom", fontsize=10)

        ax.set(xlabel="MLT",
               ylabel=r"$\mathregular{j_\parallel/\Sigma_P}$ (mA mho$^{-1}$ m$^{-1}$)")
        ax.xaxis.set_major_locator(MultipleLocator(6))

    def plot_r1_potential(self, ax):
        """Plot the electric potential in Region 1 as in Figure 2a-b."""
        ax.plot(self.mlt, self.phi_r1() / 1e3)
        ax.set(xlabel="MLT", ylabel=r"$\mathregular{\phi_{R1}}$ (kV)")
        ax.xaxis.set_major_locator(MultipleLocator(6))

    @staticmethod
    def add_cax(fig, ax, pad=0.05, width=0.01, position="right"):
        """Add a colourbar axis to a figure."""
        if isinstance(ax, np.ndarray):
            if len(ax.shape) == 1:
                ax_bbox = ax[-1].get_position()
                cax_bbox = ax_bbox

                if position == "right":
                    cax_bbox.y0 = ax[0].get_position().y0
                else:
                    cax_bbox.x0 = ax[0].get_position().x0
            else:
                ax_bbox = ax[-1, -1].get_position()
                cax_bbox = ax_bbox

                if position == "right":
                    cax_bbox.y1 = ax[0, -1].get_position().y1
                else:
                    cax_bbox.x0 = ax[-1, 0].get_position().x0
        else:
            ax_bbox = ax.get_position()
            cax_bbox = ax_bbox

        if position == "right":
            cax_bbox.x0 = ax_bbox.x1 + pad
            cax_bbox.x1 = cax_bbox.x0 + width
        elif position == "below":
            cax_bbox.y1 = ax_bbox.y0 - pad
            cax_bbox.y0 = cax_bbox.y1 - width
        else:
            raise ValueError('Position must be "right" or "below".')

        cax = fig.add_axes(cax_bbox)

        return cax

    @staticmethod
    def _annotate_map(ax, component):
        if component == "labda":
            annotation = r"$\mathregular{\lambda}$"
        else:
            annotation = r"$\mathregular{\theta}$"

        ax.annotate(annotation, xy=(1, 1), xycoords="axes fraction",
                    xytext=(-5, -5), textcoords="offset points",
                    fontsize="xx-large", ha="right", va="top")

    @staticmethod
    def _configure_map(ax, rmax, colat_grid_spacing=10, theta_range=None, mlt=True):
        """Configures a polar plot with midnight at the bottom and sensible labelling."""

        def format_mlt():
            """Return MLT in hours rather than a number of degrees when drawing axis labels."""

            # noinspection PyUnusedLocal
            def formatter_function(y, pos):
                hours = y * (12 / np.pi)
                if hours == 24:
                    return ""
                else:
                    if hours < 0:
                        hours += 24
                    return f"{hours:.0f}"

            return FuncFormatter(formatter_function)

        # Configure colatitude.
        ax.set_rmin(0.0)
        ax.set_rmax(rmax)
        ax.yaxis.set_major_locator(MultipleLocator(colat_grid_spacing))

        ax.set_theta_zero_location("S")

        if theta_range is not None:
            ax.set_thetamin(theta_range[0])
            ax.set_thetamax(theta_range[1])

        if mlt:
            ax.xaxis.set_major_formatter(format_mlt())
            ax.xaxis.set_major_locator(MultipleLocator(np.pi / 2))

        ax.grid(True)

    def _plot_map(self, ax, variable, vmin, vmax, cmap, contours, **kwargs):
        mesh = ax.pcolormesh(self.theta, self.colat, variable, cmap=cmap,
                             vmin=vmin, vmax=vmax, shading="nearest", **kwargs)

        if contours:
            self.draw_potential_contours(ax)
        self._configure_map(ax, 30)

        return mesh


class Model(BaseModel):
    def __init__(self, phi_d, phi_n, f_107, time, hemisphere, sigma_h=12, sigma_p=7,
                 precipitation_conductance="add", **kwargs):
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

        self.sza = self.sza_grid()
        self.sigma_h, self.sigma_p = self.sigma_grid(sigma_h, sigma_p)
        self.div_jp, self.div_jh = self.div_j_grid()
        self.j = self.div_jp + self.div_jh

    def sza_grid(self):
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
        """Grids of quiet-time Hall and Pedersen conductance (Moen and Brekke, 1993)."""
        sigma_h = np.zeros_like(self.sza)
        sigma_p = np.zeros_like(self.sza)

        # The contribution from anything on the nightside is 0, so only compute on the dayside.
        day_mask = self.sza < np.pi / 2.
        sza_day = self.sza[day_mask]

        sigma_h[day_mask] = (self.f_107 ** 0.53) * ((0.81 * np.cos(sza_day))
                                                    + (0.54 * np.sqrt(np.cos(sza_day))))
        sigma_p[day_mask] = (self.f_107 ** 0.49) * ((0.34 * np.cos(sza_day))
                                                    + (0.93 * np.sqrt(np.cos(sza_day))))

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
        mesh = self._plot_map(ax, np.degrees(self.sza), vmin, vmax, cmap, contours, **kwargs)
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


class DailyAverage(object):
    def __init__(self, phi_d, f_107, day, hemisphere, **kwargs):
        """
        The FACER model calculated at both UT=5 and UT=17 for the input day and then averaged. Over a timescale of one
        day the dayside and nightside reconnection rates can be assumed to be approximately equal (Cowley and Lockwood,
        1992) and so only the dayside reconnection rate need be provided.

        Parameters
        ----------
        phi_d : float
            Reconnection rates, in kV.
        f_107 : float
            The F10.7 index, in solar flux units.
        day : datetime.datetime
        hemisphere : basestring
        """
        if day.hour != 0 or day.minute != 0 or day.second != 0 or day.microsecond != 0:
            raise ValueError("The day must not have any associated time information.")

        self.ut_5 = Model(phi_d, phi_d, f_107, day + timedelta(hours=5), hemisphere, **kwargs)
        self.ut_17 = Model(phi_d, phi_d, f_107, day + timedelta(hours=17), hemisphere, **kwargs)

        self.j = np.median((self.ut_5.j_total(), self.ut_17.j_total()))
