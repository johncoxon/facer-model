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

from .full import Model
from datetime import timedelta
from numpy import median

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

        self.j = median((self.ut_5.j_total(), self.ut_17.j_total()))
