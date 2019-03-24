import numpy as np

import constants as c
from assimulo.problem import Implicit_Problem
from assimulo.solvers import Radau5DAE


class ECAR_Implicit_Problem(Implicit_Problem):
    def __init__(self, system, **kwargs):
        Implicit_Problem.__init__(self, **kwargs)
        self.name = 'ECAR Problem without concentration checks'
        self.system = system

    # residual function - function that should evaluate to zero to fulfill the conditions of the DAE
    def res(self, t, y, yd):
        spec_rhs = self.system.rhs(t, y)
        return np.array([
            spec_rhs[0], spec_rhs[1], spec_rhs[2], spec_rhs[3],
            spec_rhs[4] - yd[4], spec_rhs[5] - yd[5], spec_rhs[6] - yd[6],
            spec_rhs[7] - yd[7], spec_rhs[8] - yd[8]
        ])


# Differential algebraic system corresponding to a given ECAR system
def dae_rhs(system):
    def dae_array(t, conc):
        initial_conc = system.initial_conc
        dosage = system.dose_rate
        c_ads = c.adsorbed_species_vector(conc)
        return np.array([
            (initial_conc[0] + initial_conc[1]) -
            (conc[0] + c_ads[0] + conc[7]),  # As(III), algebraic
            conc[7] - (c_ads[1] + conc[1]),  # As(V), algebraic
            conc[2] - (initial_conc[2] - c_ads[2]),  # P, algebraic
            conc[3] - (initial_conc[3] - c_ads[3]),  # Si, algebraic
            (-1 * c.k_app(conc[8]) * conc[4] * conc[6]) +
            (int(t <= system.dose_time_sec) * dosage),  # Fe(II), gen - conversion to Fe(III)
            c.k_app(conc[8]) * conc[4] * conc[6],  # Fe(III), pure generation
            c.k_r * (c.O2_saturation_25C - conc[6]) -
            (c.k_app(conc[8]) * conc[4] * conc[6]),
            # O2, mass transfer from air - removal by Fe
            (c.beta * c.k_app(conc[8]) * conc[4] * conc[6]
             ) / (1 + (c.k_1_div_k_2(conc[8]) * conc[4] / conc[0])),
            # As(V) tot = As(III) oxidized
            0  # pH, approximate
        ])
    return dae_array


# computes derivative of initial condition
def initial_derivative(system):
    # compute DAE RHS with conc = init_conc; this gives 5th through 9th values of initial derivative
    init_dae_rhs = dae_rhs(system)(0, system.initial_conc)
    return np.array([
        0,  # initial As (III) time derivative
        0,  # initial As (V) time derivative
        0,  # initial P time derivative
        0,  # initial Si time derivative
        init_dae_rhs[4],  # time derivative of Fe(II), gen - conversion
        init_dae_rhs[5],  # time derivative of Fe(III), pure generation
        init_dae_rhs[6],  # time derivative of O2, mass transfer - removal
        init_dae_rhs[7],  # time derivative of As(V) tot = As(III) oxidized
        init_dae_rhs[8]  # time derivative of pH, approximate
    ])


def integrate(system, t0, tf):
    # set up DAE
    system.rhs = dae_rhs(system)

    system.initial_conc_td = initial_derivative(system)
    prob = ECAR_Implicit_Problem(
        system=system,
        y0=system.initial_conc,
        yd0=system.initial_conc_td,
        t0=t0)
    # set up solver
    sol = Radau5DAE(prob)
    sol.atol, sol.rtol = 1e-7, 1e-5
    return sol.simulate(tfinal=tf)
