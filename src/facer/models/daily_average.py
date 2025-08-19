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
from datetime import timedelta
from .full_model import Model

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

        self.ut = {}
        j_totals = []

        for hour in np.arange(24):
            self.ut[hour] = Model(phi_d, phi_d, f_107, day + timedelta(hours=int(hour)), hemisphere, **kwargs)
            j_totals.append(self.ut[hour].j_total())

        self.j = np.mean(j_totals)