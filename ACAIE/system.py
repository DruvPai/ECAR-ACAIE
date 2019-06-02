import numpy as np

import constants as c
import dae


class ACAIE:
    def __init__(self, dose_time, mix_time, dose_rate, dose_load,
                 initial_conc):
        self.dose_time, self.mix_time, self.dose_rate, self.dose_load, self.initial_conc = dose_time, mix_time, dose_rate, dose_load, initial_conc
        # From two of dose_time, dose_rate, dose_load, compute the third.
        if not (
                (c.approx(self.dose_time, 0) and c.approx(self.dose_rate, 0)
                 and c.approx(self.dose_load, 0)) or
                (c.approx(self.dose_time, 0) and c.approx(self.dose_rate, 0)) or
                (c.approx(self.dose_time, 0) and c.approx(self.dose_rate, 0)) or
                (c.approx(self.dose_rate, 0) and c.approx(self.dose_load, 0))):
            if c.approx(self.dose_time, 0):
                self.dose_time = self.dose_load / self.dose_rate  # [min]
            elif c.approx(self.dose_rate, 0):
                self.dose_rate = self.dose_load / self.dose_time  # [C/(L-min)]
            elif c.approx(self.dose_load, 0):
                self.dose_load = self.dose_rate * self.dose_time  # [C/L]
            elif c.approx(self.dose_rate, self.dose_load * self.dose_time
                          ):  # check for consistency if everything is set
                raise Exception('Error: inconsistent dose settings')
        else:
            self.dose_time, self.dose_rate, self.dose_load = 0, 0, 0
        # Change dose_rate into M/s units
        self.fe_dosage_rate = self.dose_rate / (2 * c.faraday_constant * 60
                                                )  # Fe dosage rate [M/s]
        self.fe_dosage_load = self.fe_dosage_rate * self.dose_time * \
            60 * c.mm_Fe * 1000  # Fe dosage load [mg/L]
        # change dosage into seconds
        self.dose_time_sec = self.dose_time * 60  # [s]
        self.mix_time_sec = self.mix_time * 60  # equilibrium mixing [s]

    def model(self):
        def conv_to_array(lst): return np.array(lst)
        if not c.approx(self.dose_time, 0):  # cases with initial dosing
            self.time_list, self.conc_list, self.conc_td_list = tuple(map(conv_to_array, dae.integrate(
                self, 0, self.dose_time_sec + self.mix_time_sec)))
            # we could binary search for the index but... what's the point? n is small
            i = max([k for k in range(self.time_list.size)
                     if self.time_list[k] < self.dose_time_sec])
            self.time_dose_list, self.conc_dose_list, self.conc_td_dose_list = self.time_list[
                :i], self.conc_list[:i], self.conc_td_list[:i]
        else:  # set dosing = 0
            self.time_list, self.conc_list, self.conc_td_list = tuple(
                map(conv_to_array, dae.integrate(self, 0, self.mix_time_sec)))
        return self.time_list, self.conc_list, self.conc_td_list
