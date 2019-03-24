import numpy as np

# Constants
faraday_constant = 96485  # Faraday constant, C/(mol e-)
O2_saturation_20C = (9.1 / 1000) / 32  # O2 saturation concentration at 20 C, M
O2_saturation_25C = 2.5 * pow(10, -4)  # O2 saturation concentration at 25 C, M

# Molecular masses, g/mol
mm_As = 74.9216  # arsenic molar mass
mm_P = 30.9738  # phosphorus molar mass
mm_Si = 28.0855  # silicon molar mass
mm_Fe = 55.845  # iron molar mass
mm_H2O2 = 34.0147  # hydrogen peroxide molar mass
mm_O2 = 31.998  # oxygen molar mass

# Rate and equilibrium coefficients
# pH range: 6.6 - 8.1
K_AsIII = pow(10,
              3.81)  # As(III) equilibrium constant, low variance in pH range
K_AsV = lambda pH: pow(10, -0.53 * pH + 8.95)  # As(V) equilibrium constant
K_P = lambda pH: pow(10, -0.65 * pH + 10.30)  # P equilibrium constant
K_Si = pow(10, 2.94)  # Si equilibrium constant, low variance in pH range
q_max = lambda pH: 0.45 * pH - 2.19  # max adsorption capacity
k_app = lambda pH: pow(10, 1.64 * pH - 10.58)  # Fe(II) oxidation by O2 rate constant
k_1_div_k_2 = lambda pH: 1.15 * pH - 6.63  # ratio of rate constants for oxidizing arsenic with oxygen
k_r = 2 / 3600  # rate constant of oxygen recharge in the system
beta = 0.25  # yield of reactive intermediate Fe(IV) during oxidation

# Helper functions for adsorbed species equilibrium concentration vector
# Dissolved concentration vector: conc_dissolved = [AsIII, AsV, P, Si, FeII, FeIII, O2, AsVtot, pH]
adsorbed_species_denom = lambda vec: 1 + K_AsIII * vec[0] + K_AsV(vec[8]) * vec[1] + K_P(vec[8]) * vec[2] + K_Si * vec[3]
adsorbed_species_numer = lambda vec: np.array([K_AsIII * vec[0], K_AsV(vec[8]) * vec[1],
                                                             K_P(vec[8]) * vec[2], K_Si * vec[3]])
# Adsorbed species volumetric equilibrium concentrations
adsorbed_species_vector = lambda vec: np.array([term * q_max(vec[8]) * vec[5] / adsorbed_species_denom(vec)
                                                for term in adsorbed_species_numer(vec)])

# Helper function to determine approximate equality
approx = lambda float1, float2: abs(float1 - float2) <= pow(10, -5)
