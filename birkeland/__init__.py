import numpy as np
from pandas import DataFrame


class Model(object):
    def __init__(self, phi_d, phi_n, f_pc, theta_d=30, theta_n=30, sigma_pc=1, sigma_rf=1,
                 delta_colat=10, r1_colat=None, order_n=20, use_igrf=False, datetime=None):
        """
        A Python implementation of the Birkeland current model presented by Milan (2013).

        A "simple mathematical model of the region 1 and 2 Birkeland current system intensities for
        differing dayside and nightside magnetic reconnection rates, consistent with the
        expanding/contracting polar cap paradigm of solar wind-magnetosphere-ionosphere coupling."

        Parameters
        ----------
        phi_d, phi_n : float
            Dayside and nightside reconnection rates, in kV.
        f_pc : float
            The polar cap flux, in GWb.
        theta_d, theta_n : float, optional, default 30
            The azimuthal widths of the dayside and nightside merging gaps, in degrees.
        sigma_pc, sigma_rf : float, optional, default 1
            The conductivities in polar cap and return flow regions, in mho.
        delta_colat : float, optional, default 10
            The colatitudinal gap between the R1 and R2 current ovals in degrees.
        r1_colat : float, optional, default None
            Set this to specify an R1 colatitude instead of calculating it directly from the polar
            cap flux F_PC. (This is implemented to allow comparisons to the original IDL.)
        order_n : int, optional, default 20
            Set this to govern the order of the Fourier terms in the model.
        use_igrf : bool, optional, default False
            Set this to True if you want to use the IGRF model instead of Equation 6 to get radial
            magnetic field.
        datetime : datetime.datetime, optional, default None
            Set this to a datetime if use_igrf is set.
        """
        self.phi_d = phi_d * 1e3
        self.phi_n = phi_n * 1e3                        # Convert to SI units from inputs
        self.f_pc = f_pc * 1e9
        self.theta_d = np.radians(theta_d)              # theta is MLT
        self.theta_n = np.radians(theta_n)
        self.sigma_pc = sigma_pc
        self.sigma_rf = sigma_rf

        # Define the ratio of conductivities (it makes the maths for R1 current easier).
        self._alpha = self.sigma_rf / self.sigma_pc

        # Configure magnetic field information for the model.
        self.use_igrf = use_igrf
        self.datetime = datetime
        self._r_e = 6.371e6                     # Earth radius of 6371 km.
        self._b_eq = 31000e-9                   # Equatorial field strength of 31,000 nT.

        # Configure the default grid for the model. Milan (2013) uses the symbol lambda to refer to
        # colatitude, but lambda has an inbuilt meaning in Python, so we use "labda" instead.
        self._n_labda = 31
        self._n_theta = 360
        self.labda = np.radians(np.linspace(0.5, 30.5, self._n_labda))   # 1° latitudinal resolution
        self.theta = np.radians(np.linspace(0.5, 359.5, self._n_theta))  # 1° azimuthal resolution
        self.colat = np.degrees(self.labda)
        self.mlt = np.degrees(self.theta) / 15

        # Calculate the colatitude of the R1 and R2 current ovals. If a colatitude has been
        # specified for R1, do not calculate the colatitude from the underlying F_PC value.
        if r1_colat:
            print("Manually overriding R1 current oval colatitude...")
            self.labda_r1 = np.radians(r1_colat)
        else:
            self.labda_r1 = self.lambda_r1()
        self.labda_r2 = self.labda_r1 + np.radians(delta_colat)

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

    def lambda_r1(self):
        """lambda_R1 determined using F_PC from the inverse of Equation 8."""
        return np.sqrt(np.arcsin(self.f_pc / (2 * np.pi * (self._r_e ** 2) * self._b_eq)))

    def v_r1(self):
        """R1 current oval velocity V_R1 from Equation 9."""
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

        # This actually does the maths from Table 1 and puts it in the table. There isn't a
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

        # Every solution from Table 1 is multiplied by -R_E * np.sin(lambda_R1), so do that now.
        phi_r1 *= -self._r_e * np.sin(self.labda_r1)

        if phi_r1.shape[0] == 1:
            phi_r1 = phi_r1[0]

        return phi_r1

    @staticmethod
    def _lambda(labda):
        """The lambda term defined at the bottom of column 1 on page 5, used in Equations 17-19."""
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

    @staticmethod
    def _coth(x):
        """Used in Equations 28 and 30."""
        return np.cosh(x) / np.sinh(x)

    @staticmethod
    def _csch(x):
        """Used in Equations 29 and 31."""
        return 1 / np.sinh(x)


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
