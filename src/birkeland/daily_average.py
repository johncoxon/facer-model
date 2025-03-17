import numpy as np
from .model import Model

class DailyAverage(object):
    def __init__(self, phi_d, f_107, day, hemisphere, **kwargs):
        """
        The expanded Milan (2013) model calculated at both UT=5 and UT=17 for the input day and then averaged.
        This simplification means that the dayside and nightside reconnection rates can be assumed to be approximately
        equal and so only the dayside reconnection rate need be provided.

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
