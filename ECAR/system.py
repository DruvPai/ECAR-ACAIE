import matplotlib.pyplot as plt
import numpy as np

import constants as c
import dae


class ECAR:
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
        self.fe_dosage_load = self.fe_dosage_rate * self.dose_time * 60 * c.mm_Fe * 1000  # Fe dosage load [mg/L]
        # change dosage into seconds
        self.dose_time_sec = self.dose_time * 60  # [s]
        self.mix_time_sec = self.mix_time * 60  # equilibrium mixing [s]

    def model(self):
        conv_to_array = lambda lst: np.array(lst)
        if not c.approx(self.dose_time, 0):  # cases with initial dosing
            self.time_list, self.conc_list, self.conc_td_list = tuple(map(conv_to_array, dae.integrate(
                self, 0, self.dose_time_sec + self.mix_time_sec)))
            # we could binary search for the index but... what's the point? n is small
            i = max([k for k in range(self.time_list.size) if self.time_list[k] < self.dose_time_sec])
            self.time_dose_list, self.conc_dose_list, self.conc_td_dose_list = self.time_list[:i], self.conc_list[:i], self.conc_td_list[:i]
        else:  # set dosing = 0
            self.time_list, self.conc_list, self.conc_td_list = tuple(
                map(conv_to_array, dae.integrate(self, 0, self.mix_time_sec)))
        return self.time_list, self.conc_list, self.conc_td_list

    # make 3 plots: dose load vs frac of conc remaining (dose), dose load vs As_tot (dose), time vs frac of conc remaining (both)
    def plot(self):
        # matplotlib helper function
        plotter = lambda ax, x, y, color: ax.plot(x, y, {'color':
                                                             color,
                                                         'marker': 'o',
                                                         'markersize': '1',
                                                         'linestyle':
                                                             'solid'})

        if not c.approx(self.dose_time,
                        0):  # use for a system with both dosing and mixing
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            time_dose_adj = np.array([
                self.dose_rate * self.time_dose_list[i]
                for i in range(self.time_dose_list.size)
            ])

            self.conc_frac_dose_list = np.array(
                [conc / self.initial_conc for conc in self.conc_dose_list])
            # make first plot
            as3_dissolved_dose_list = self.conc_frac_dose_list[:, 0]
            as3_tot_dose_list = as3_dissolved_dose_list + np.array([
                c.adsorbed_species_vector(self.conc_dose_list[i])[0] /
                self.initial_conc[0] for i in range(self.conc_dose_list.size)
            ])
            p_dose_list = self.conc_frac_dose_list[:, 2]
            si_dose_list = self.conc_frac_dose_list[:, 3]
            as5_tot_dose_list = self.conc_frac_dose_list[:, 7]

            plotter(ax1, time_dose_adj, as3_dissolved_dose_list, 'b')
            plotter(ax1, time_dose_adj, as3_tot_dose_list, 'r')
            plotter(ax1, time_dose_adj, as5_tot_dose_list, 'g')
            plotter(ax1, time_dose_adj, p_dose_list, 'c')
            plotter(ax1, time_dose_adj, si_dose_list, 'm')
            ax1.title(
                'Fractions of initial concentrations v.s. dose load (mg/L)')
            ax1.ylabel('Fraction of initial concentrations')
            # make second plot
            as_tot_dose_list = np.array(
                [(self.conc_dose_list[0] + self.conc_dose_list[7] +
                  c.adsorbed_species_vector(self.conc_dose_list[i])[0]) /
                 (self.initial_conc[0] + self.initial_conc[7])
                 for i in range(self.conc_dose_list.size)])
            plotter(ax2, time_dose_adj, as_tot_dose_list, 'b')
            ax2.ylabel('Fraction of initial total arsenic')
            ax2.xlabel('Dose load (Fe, mg/L)')
        else:
            fig, ax3 = plt.subplots(1, 1)

        # make third plot
        self.conc_frac_list = [conc / self.initial_conc for conc in self.conc_list]
        as3_dissolved_list = self.conc_frac_list[:, 0]
        as3_tot_list = as3_dissolved_list + np.array([
            c.adsorbed_species_vector(self.conc_list[i])[0] / self.initial_conc[0]
            for i in range(self.conc_list.size)
        ])
        p_list = self.conc_frac_list[:, 2]
        si_list = self.conc_frac_list[:, 3]
        as5_tot_list = self.conc_frac_list[:, 7]
        plotter(ax3, self.time_list, as3_dissolved_list, 'b')
        plotter(ax3, self.time_list, as3_tot_list, 'r')
        plotter(ax3, self.time_list, as5_tot_list, 'g')
        plotter(ax3, self.time_list, p_list, 'c')
        plotter(ax3, self.time_list, si_list, 'm')
        ax3.title('Fractions of initial concentrations v.s. time (s)')
        ax3.pyplot.ylabel('Fraction of initial concentration')
        ax3.pyplot.xlabel('Time (s)')
        plt.show()
