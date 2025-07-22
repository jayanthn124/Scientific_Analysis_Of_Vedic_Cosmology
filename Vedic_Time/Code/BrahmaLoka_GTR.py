import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Constants
G = 6.67430e-11       # Gravitational constant (m^3 kg^-1 s^-2)
c = 299792458         # Speed of light (m/s)
pi = np.pi

# Target values
gamma_target = 3.154e12
T_target = 9.8156e19  # seconds

# Kerr spin parameter (maximal spin: a = GM/c^2)
def kerr_spin(M):
    return G * M / c**2

# Orbital velocity for circular orbit
def orbital_velocity(M, r):
    return np.sqrt(G * M / r)

# Gravitational time dilation factor in equatorial circular orbit (approximate)
def gamma_kerr(M, r):
    a = kerr_spin(M)
    v = orbital_velocity(M, r)
    term = 1 - (3 * G * M) / (r * c**2) + (2 * a * v) / (r * c**2)
    if term <= 0 or np.isnan(term):
        return np.inf
    return 1 / np.sqrt(term)

# Orbital period in Boyer-Lindquist coordinates (approximate)
def orbital_period(M, r):
    a = kerr_spin(M)
    base = 2 * pi * np.sqrt(r**3 / (G * M))
    correction = 1 + (a * np.sqrt(G * M)) / (r**(3/2) * c)
    return base / correction

# Approximate ISCO radius for prograde orbit (ensure stability)
def isco_radius(M):
    return kerr_spin(M) + 3 * G * M / c**2  # Conservative lower bound

# Error function in log space for stability
def log_error(M_log10):
    M = 10**M_log10
    r_min = isco_radius(M)
    r_vals = np.logspace(np.log10(r_min * 1.01), 30, 500)
    
    min_error = np.inf
    best_result = None

    with np.errstate(all='ignore'):
        for r in r_vals:
            g = gamma_kerr(M, r)
            T = orbital_period(M, r)

            if g == np.inf or T == np.inf or np.isnan(g) or np.isnan(T):
                continue

            g_error = np.log10(g / gamma_target)**2
            T_error = np.log10(T / T_target)**2
            total_error = g_error + T_error

            if total_error < min_error:
                min_error = total_error
                best_result = (M, r, g, T, total_error)

    if best_result:
        M_best, r_best, g_best, T_best, err = best_result
        print(f"Candidate Match:\n M = {M_best:.3e} kg\n r = {r_best:.3e} m\n γ = {g_best:.3e}\n T = {T_best:.3e} s\n Error = {err:.3e}")
        return err
    else:
        return np.inf

# Optimization over log10(M)
result = minimize_scalar(log_error, bounds=(30, 60), method='bounded')

if result.success:
    M_opt = 10**result.x
    r_min = isco_radius(M_opt)
    r_vals = np.logspace(np.log10(r_min * 1.01), 30, 500)

    min_error = np.inf
    best_config = None

    for r in r_vals:
        g = gamma_kerr(M_opt, r)
        T = orbital_period(M_opt, r)

        if g == np.inf or T == np.inf or np.isnan(g) or np.isnan(T):
            continue

        g_error = np.log10(g / gamma_target)**2
        T_error = np.log10(T / T_target)**2
        total_error = g_error + T_error

        if total_error < min_error:
            min_error = total_error
            best_config = (M_opt, r, g, T, total_error)

    if best_config:
        M_best, r_best, g_best, T_best, err = best_config
        print("\n=== Best Configuration Found ===")
        print(f"Black Hole Mass (M):     {M_best:.3e} kg")
        print(f"Orbital Radius (r):      {r_best:.3e} m")
        print(f"Time Dilation Factor (Computed): {g_best:.6e}")
        print(f"Orbital Period (T):      {T_best:.6e} s")
        print(f"Total Log Error:         {err:.3e}")
    else:
        print("Failed to find a valid orbital configuration.")
else:
    print("Optimization over M failed.")