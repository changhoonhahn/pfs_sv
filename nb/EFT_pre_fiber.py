# ============================================================
# Baseline full-shape correlation-function analysis before fibre assignment
# ============================================================
#
# Purpose:
#   This script generates a theoretical mock two-point correlation function
#   using a fiducial cosmology and fits it with a full-shape LPT/Velocileptors
#   model. Unlike the post-fibre-assignment script, this version does not apply
#   a post/pre fibre-assignment ratio to the mock data.
#
# Main workflow:
#   1. Define the fiducial cosmology and analysis scale range.
#   2. Use CLASS to find the primordial amplitude A_s corresponding to sigma8 = 0.8.
#   3. Generate mock correlation-function multipoles xi_0 and xi_2.
#   4. Use the unmodified mock multipoles as the data vector.
#   5. Load the pre-fibre-assignment covariance matrix.
#   6. Build a Gaussian likelihood with desilike.
#   7. Build a Taylor emulator for fast theory evaluation.
#   8. Run MCMC sampling.
#   9. Compute derived parameters sigma8 and Omega_m from the chain.
#  10. Refine the best-fit point by direct chi^2 minimization.

# Recommend running this code with Python 3.11. 
# conda install conda-forge::numpy  
# conda install conda-forge::scipy  
# conda install conda-forge::pyyaml  
# conda install anaconda::mpi4py  
# python -m pip install git+https://github.com/cosmodesi/cosmoprimo (https://cosmoprimo.readthedocs.io/en/latest/user/building.html)  
# python -m pip install git+https://github.com/adematti/pyclass (https://github.com/adematti/pyclass)  
# python -m pip install git+https://github.com/cosmodesi/desilike#egg=desilike[plotting,jax] (https://github.com/cosmodesi/desilike)  
# python -m pip install git+https://github.com/cosmodesi/pypower (https://github.com/cosmodesi/pypower)  
# conda install -c conda-forge zeus-mcmc (https://github.com/minaskar/zeus/tree/main)  
# conda install -c conda-forge "numpy<2.0"
# conda install -c conda-forge pyfftw (https://github.com/hgomersall/pyFFTW)
# python3 -m pip install -v git+https://github.com/sfschen/velocileptors (https://github.com/sfschen/velocileptors/tree/master)
# pip install classy (https://github.com/lesgourg/class_public/wiki/Installation)


import numpy as np
from matplotlib import pyplot as plt
from cosmoprimo import Cosmology
from pypower.fft_window import get_correlation_function_tophat_derivative
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from pypower import CorrelationFunctionStatistics
from desilike.theories.galaxy_clustering import (
    DirectPowerSpectrumTemplate,
    LPTVelocileptorsTracerCorrelationFunctionMultipoles,
)
from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.parameter import ParameterCollection
from desilike import setup_logging
from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine
import os
from desilike.samplers import ZeusSampler
from desilike.samples import plotting
from classy import Class
from tqdm import tqdm
from getdist import MCSamples, plots
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import minimize


# Effective redshift of the mock sample.
z = 0.15

# Separation-bin edges for the correlation function.
# The fitting range is 40--160 Mpc/h, divided into 20 bins.
sedges = np.linspace(40., 160., 21)

# Separation-bin centers used for theoretical predictions.
s = (sedges[:-1] + sedges[1:]) / 2.
ns = s.size

# k-bin edges used only for estimating the number of Fourier modes.
# These are stored in the CorrelationFunctionStatistics object.
kedges = np.linspace(0., 0.3, 41)
nmodes = 4. * np.pi / 3. * (kedges[1:]**3 - kedges[:-1]**3)


# Fiducial cosmology used to generate the mock correlation function.
h = 0.71
omega_b = 0.02258
Omega_m = 0.2648
n_s = 0.963
sigma8 = 0.8

# Convert Omega_m into the physical cold dark matter density.
# omega_cdm = Omega_cdm h^2 = Omega_m h^2 - omega_b.
omega_cdm = Omega_m * h**2 - omega_b


# ============================================================
# Find A_s corresponding to the target sigma8 using CLASS
# ============================================================

# CLASS takes A_s as an input parameter, while the analysis target is sigma8.
# We therefore sample several A_s values, compute sigma8 for each, and fit an
# approximate relation A_s(sigma8).
As_samples = np.linspace(2.0e-9, 2.3e-9, 20)
sigma8_samples = []

print("Running CLASS to gather data points...")

cosmo = Class()

for A_s in As_samples:
    params = {
        'output': 'mPk',
        'h': h,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'n_s': n_s,
        'A_s': A_s,
        'P_k_max_h/Mpc': 1.0,
        'z_pk': 0.,
    }

    # Compute sigma8 for the current A_s value.
    cosmo.set(params)
    cosmo.compute()
    sigma8_samples.append(cosmo.sigma8())

    # Clean CLASS internal structures before the next evaluation.
    cosmo.struct_cleanup()

As_array = np.array(As_samples)
s8_array = np.array(sigma8_samples)

# Fit A_s as a quadratic function of sigma8.
# This is sufficient here because the sampled range is narrow.
coeffs = np.polyfit(s8_array, As_array, 2)
poly_func = np.poly1d(coeffs)

# Interpolate the A_s value corresponding to the target sigma8.
target_s8 = 0.8
target_As = poly_func(target_s8)
logA_target = np.log(1e10 * target_As)

# Create the fiducial cosmology object used by cosmoprimo/desilike.
fiducial_cosmo = Cosmology(
    h=h,
    omega_b=omega_b,
    n_s=n_s,
    omega_cdm=omega_cdm,
    A_s=target_As,
    Omega_k=0.0,
    engine='class',
)

# desilike uses logA = ln(10^10 A_s).
logA_val = np.log(10**10 * fiducial_cosmo.A_s)
print(f"Calculated logA = {logA_val}")


# ============================================================
# Generate the baseline mock correlation-function multipoles
# ============================================================

# Build the linear power-spectrum template at the effective redshift.
template = DirectPowerSpectrumTemplate(z=z, fiducial=fiducial_cosmo)

# Explicitly set the fiducial cosmological parameters in the template.
template.params.update({
    'h': h,
    'omega_b': omega_b,
    'omega_cdm': omega_cdm,
    'n_s': n_s,
    'logA': logA_val,
})

# Build the LPT/Velocileptors model for correlation-function multipoles.
# The model returns xi_0 and xi_2 evaluated at the separation-bin centers.
theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(
    template=template,
    s=s,
    ells=(0, 2),
)

# Mock parameters used to generate the baseline data vector.
# These values should be recovered by the fit if the pipeline is internally consistent.
mock_params = {
    # Lagrangian bias parameters.
    'b1p': 1.1327562,
    'b2p': -0.34870614,
    'bsp': -0.25395155,
    'b3p': 0.0,

    # EFT counterterms.
    'alpha0p': 0.1,
    'alpha2p': 0.1,
    'alpha4p': 0.0,
    'alpha6p': 0.0,
}

# Evaluate the theoretical model at the mock parameter values.
theory(**mock_params)
print("LPT calculation computed.")

# Store the generated mock multipoles.
poles = theory.corr

# Shot noise corresponding to an assumed number density nbar = 5e-4.
shotnoise = 1 / 5e-4

# Wrap the mock multipoles in a pypower CorrelationFunctionStatistics object.
# This object is the data format expected by the desilike observable.
mean = CorrelationFunctionStatistics(
    sedges,
    s,
    poles,
    nmodes=nmodes,
    ells=(0, 2),
    shotnoise_nonorm=shotnoise,
    statistic='multipole',
)

print("\n--- Mock Data (Multipoles) ---")
print("s:", s[:5])
print("xi0:", mean.corr[0][:5])  # Monopole xi_0.
print("xi2:", mean.corr[1][:5])  # Quadrupole xi_2.


# This baseline script uses the pre-fibre-assignment covariance matrix.
# The covariance must match the ordering and dimension of the flattened data vector.
cov_path = 'cov_s_pre.npy'
try:
    cov = np.load(cov_path)
    print(f"\nCovariance matrix loaded from {cov_path}")
except FileNotFoundError:
    print(f"\nWarning: Covariance file not found at {cov_path}. 'cov' variable is not set.")
    cov = None

# In this baseline run, the data vector is the unmodified theoretical mock.
# No fibre-assignment post/pre ratio is applied here.
data = mean


# ============================================================
# Define the fitting model and parameter priors
# ============================================================

z = 0.15
template = DirectPowerSpectrumTemplate(z=z, fiducial=fiducial_cosmo)

# Fix selected cosmological template parameters.
# Note: later parts of the script still export or pass omega_b and n_s.
# If strict fixed/free behavior matters, inspect the final desilike parameter table.
for param in ['omega_b', 'n_s']:
    template.params[param].update(fixed=True)

# Theory calculator used in the likelihood.
theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(template=template)

# Define priors and fixed values for bias and EFT parameters.
theory.params.update({
    # ------------------------------------------------------------
    # 1. Bias parameters
    # ------------------------------------------------------------
    # Prior motivation:
    #   The comment in the original code suggests
    #       (1 + b1) * sigma8 ~ Uniform(0.5, 3.0)
    #   which gives approximately
    #       b1 ~ Uniform(-0.375, 2.75)
    #   for sigma8 = 0.8.
    'b1p': {
        'prior': {'dist': 'uniform', 'limits': [-0.375, 2.75]},
        'ref': 1.1327562,
        'proposal': 0.05,
    },

    # Second-order bias parameter with a broad Gaussian prior.
    'b2p': {
        'prior': {'dist': 'norm', 'loc': 0.0, 'scale': np.sqrt(10)},
        'ref': -0.34870614,
        'proposal': 0.5,
    },

    # Tidal bias parameter with a broad Gaussian prior.
    'bsp': {
        'prior': {'dist': 'norm', 'loc': 0.0, 'scale': np.sqrt(5)},
        'ref': -0.25395155,
        'proposal': 0.5,
    },

    # b3p is fixed to zero in this analysis.
    'b3p': {'value': 0.0, 'fixed': True},

    # ------------------------------------------------------------
    # 2. EFT counterterms
    # ------------------------------------------------------------
    # alpha0p and alpha2p are allowed to vary with broad Gaussian priors.
    'alpha0p': {
        'prior': {'dist': 'norm', 'loc': 0.0, 'scale': 10.0},
        'ref': 0.1,
        'proposal': 1.0,
    },

    'alpha2p': {
        'prior': {'dist': 'norm', 'loc': 0.0, 'scale': 10.0},
        'ref': 0.1,
        'proposal': 1.0,
    },

    # Higher-order counterterms are fixed to zero.
    'alpha4p': {'value': 0.0, 'fixed': True},
    'alpha6p': {'value': 0.0, 'fixed': True},
})


# ============================================================
# Build observable and Gaussian likelihood
# ============================================================

# The observable compares the baseline mock data with the LPT/Velocileptors theory.
# Both monopole and quadrupole are fitted over 40--160 Mpc/h.
observable = TracerCorrelationFunctionMultipolesObservable(
    data=data,
    covariance=cov,
    slim={0: (40., 160.), 2: (40., 160.)},
    ells=(0, 2),
    theory=theory,
)

# Gaussian likelihood:
#   chi^2 = (data - theory)^T C^{-1} (data - theory)
likelihood = ObservablesGaussianLikelihood(observables=[observable])

setup_logging()

# Initial call to initialize likelihood internals.
likelihood()


# ============================================================
# Build and save a Taylor emulator
# ============================================================

# The exact LPT/Velocileptors calculation can be expensive during MCMC.
# A Taylor emulator accelerates repeated likelihood evaluations.
emulator = Emulator(
    theory,
    engine=TaylorEmulatorEngine(order={'*': 2, 'sn0': 1}),
)

# Generate emulator training samples and fit the Taylor expansion.
emulator.set_samples()
emulator.fit()
# emulator.plot(name=None)

# Save the emulator to disk for reproducibility or later reuse.
base_dir = '_tests/EFT_pre_check'
EFT_pre_emulator_fn_cosmo = os.path.join(base_dir, 'emulator_13.npy')
emulator.save(EFT_pre_emulator_fn_cosmo)

# Replace the exact theory calculator with the emulator calculator.
observable.init.update(theory=emulator.to_calculator())


# ============================================================
# Run MCMC sampling
# ============================================================

# Sample the posterior distribution using ZeusSampler.
sampler = ZeusSampler(
    likelihood,
    save_fn='_tests/EFT_pre_check/chain_fs_direct_13_*.npy',
    seed=42,
)

# Run until the convergence criterion is satisfied.
sampler.run(check={'max_eigen_gr': 0.1})


# ============================================================
# Diagnostic likelihood evaluation at the mock input point
# ============================================================

# Evaluate the likelihood at the same parameters used to generate the mock.
# For a perfectly consistent baseline mock, this should give a very small chi^2,
# up to numerical, covariance, and emulator errors.
check_params = {
    'h': h,
    'omega_b': omega_b,
    'omega_cdm': omega_cdm,
    'n_s': n_s,
    'logA': logA_val,
    'b1p': mock_params.get('b1p', 1.1327562),
    'b2p': mock_params.get('b2p', -0.34870614),
    'bsp': mock_params.get('bsp', -0.25395155),
    'b3p': mock_params.get('b3p', 0.0),
    'alpha0p': mock_params.get('alpha0p', 0.1),
    'alpha2p': mock_params.get('alpha2p', 0.1),
    'alpha4p': mock_params.get('alpha4p', 0.0),
    'alpha6p': mock_params.get('alpha6p', 0.0),
}

try:
    log_posterior = likelihood(**check_params)

    # desilike versions may expose chi2 directly or only loglikelihood.
    if hasattr(likelihood, 'chi2'):
        chi2_val = likelihood.chi2
    else:
        chi2_val = -2 * likelihood.loglikelihood

    print(f"Result:")
    print(f"  Chi^2 value    : {chi2_val:.5f}")
    print(f"------------------------------------------------")

    obs = likelihood.observables[0]

    # If available, inspect the residual between the data vector and model vector.
    if hasattr(obs, 'flatdata') and hasattr(obs, 'flattheory'):
        data_vec = obs.flatdata
        model_vec = obs.flattheory
        diff_vec = data_vec - model_vec

        max_diff = np.max(np.abs(diff_vec))
        print(f"  Max residual (Data - Model): {max_diff:.5e}")

    else:
        print("  Warning: Could not extract data/theory vectors for residual check.")

    if chi2_val < 0.1:
        print("  Status: Excellent match (Zero Chi^2 expected for Mock).")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nError: {e}")


# ============================================================
# Remove burn-in and summarize the MCMC chain
# ============================================================

# Remove the first half of the chain as burn-in.
chain = sampler.chains[0].remove_burnin(0.5)
print(chain.to_stats(tablefmt='pretty'))


# ============================================================
# Compute sigma8 for every chain sample using CLASS
# ============================================================

# The sampled parameter is logA, but sigma8 is easier to interpret physically.
# Therefore, we recompute sigma8 for every MCMC sample using CLASS.
cosmo = Class()

n_walkers, n_steps = chain['h'].shape
sigma8_chain = np.zeros((n_walkers, n_steps))

print(f"Starting calculation for {n_walkers * n_steps} points...")

for i in range(n_walkers):
    for j in tqdm(range(n_steps), desc=f"Walker {i+1}/{n_walkers}"):

        cosmo.set({
            'output': 'mPk',
            # 'non_linear': 'HALOFIT',
            'Omega_cdm': chain['omega_cdm'][i, j].item() / (chain['h'][i, j].item()**2),
            'Omega_b': chain['omega_b'][i, j].item() / (chain['h'][i, j].item()**2),
            'h': chain['h'][i, j].item(),
            'n_s': chain['n_s'][i, j].item(),
            'A_s': 1e-10 * np.exp(chain['logA'][i, j].item()),
            'z_pk': 0.,
            'P_k_max_h/Mpc': 0.5,
        })

        cosmo.compute()
        sigma8_chain[i, j] = cosmo.sigma8()

        # Clean CLASS internal structures after each evaluation.
        cosmo.struct_cleanup()

print("Calculation complete.")
print(f"Mean Sigma8: {np.mean(sigma8_chain)}")


# ============================================================
# Compute Omega_m and export MCMC samples to CSV
# ============================================================

# Derived matter density parameter:
#   Omega_m = (omega_cdm + omega_b) / h^2
Omega_m_chain = (chain['omega_cdm'] + chain['omega_b']) / (chain['h']**2)

print("Preparing data for CSV export, including chi^2 and Omega_m...")

# Flatten the walker and step dimensions for CSV output.
h_flat = chain['h'].flatten()
omega_cdm_flat = chain['omega_cdm'].flatten()
omega_b_flat = chain['omega_b'].flatten()

omegam_flat = (omega_cdm_flat + omega_b_flat) / (h_flat**2)
sigma8_flat = sigma8_chain.flatten()

# Convert log-likelihood to chi^2.
logl_flat = chain['loglikelihood'].flatten()
chi2_flat = -2.0 * logl_flat

# Store sampled and derived quantities.
data_dict = {
    'h': h_flat,
    'omega_cdm': omega_cdm_flat,
    'omega_b': omega_b_flat,
    'n_s': chain['n_s'].flatten(),
    'logA': chain['logA'].flatten(),
    'b1p': chain['b1p'].flatten(),
    'b2p': chain['b2p'].flatten(),
    'bsp': chain['bsp'].flatten(),
    'alpha0p': chain['alpha0p'].flatten(),
    'alpha2p': chain['alpha2p'].flatten(),
    'sigma8': sigma8_flat,
    'Omega_m': omegam_flat,
    'chi2': chi2_flat,
}

# Convert to a pandas DataFrame and save to CSV.
df = pd.DataFrame(data_dict)
output_csv = "mcmc_chain_results_with_chi2_minimize_1.csv"
df.to_csv(output_csv, index=False)

print(f"Success. Chain data, Omega_m, and chi^2 saved to {output_csv}. Total samples: {len(df)}")


# ============================================================
# Direct minimization of chi^2 using the likelihood
# ============================================================

print("\n================================================")
print("Starting direct chi^2 minimization using likelihood")
print("================================================")

# Use the MCMC sample with the lowest chi^2 as the starting point for optimization.
best_idx = df["chi2"].idxmin()
best_row = df.loc[best_idx]

print("\n--- MCMC best point used as optimizer start ---")
print(f"Row Index    : {best_idx}")
print(f"chi^2        : {best_row['chi2']:.8f}")
print(f"sigma_8      : {best_row['sigma8']:.8f}")
print(f"Omega_m      : {best_row['Omega_m']:.8f}")

# Parameters varied by the deterministic optimizer.
param_names = [
    "h",
    "omega_cdm",
    "omega_b",
    "n_s",
    "logA",
    "b1p",
    "b2p",
    "bsp",
    "alpha0p",
    "alpha2p",
]

# Initial point for optimization: best MCMC sample.
x0 = np.array([best_row[p] for p in param_names], dtype=float)

# Track optimization progress.
eval_counter = {"n": 0}
best_seen = {"chi2": np.inf, "x": None}


def chi2_func(x):
    """
    Objective function for direct chi^2 minimization.

    Parameters
    ----------
    x : ndarray
        Parameter vector ordered according to param_names.

    Returns
    -------
    chi2_val : float
        chi^2 value computed by the desilike likelihood.
    """
    eval_counter["n"] += 1

    # Convert optimizer vector to a parameter dictionary.
    params = dict(zip(param_names, x))

    # Fixed parameters in the theory/likelihood setup.
    params["b3p"] = 0.0
    params["alpha4p"] = 0.0
    params["alpha6p"] = 0.0

    try:
        likelihood(**params)

        if hasattr(likelihood, "chi2"):
            chi2_val = float(likelihood.chi2)
        else:
            chi2_val = float(-2.0 * likelihood.loglikelihood)

        # Penalize invalid likelihood evaluations.
        if not np.isfinite(chi2_val):
            return 1e30

        # Print whenever the optimizer finds a better point.
        if chi2_val < best_seen["chi2"]:
            best_seen["chi2"] = chi2_val
            best_seen["x"] = np.array(x, copy=True)
            print(f"[eval {eval_counter['n']:5d}] new best chi2 = {chi2_val:.8f}")

        return chi2_val

    except Exception as e:
        # Return a very large chi^2 when evaluation fails.
        print(f"[eval {eval_counter['n']:5d}] FAILED: {repr(e)}")
        return 1e30


# Evaluate the starting point.
chi2_x0 = chi2_func(x0)
print(f"\nchi2(x0) = {chi2_x0:.8f}")

# Bounds for L-BFGS-B optimization.
# They are broad but chosen to remain in a physically stable region.
bounds = [
    (0.5, 0.9),      # h
    (0.05, 0.2),     # omega_cdm
    (0.018, 0.03),   # omega_b
    (0.9, 1.05),     # n_s
    (2.5, 3.5),      # logA
    (-0.375, 2.75),  # b1p
    (-10.0, 10.0),   # b2p
    (-10.0, 10.0),   # bsp
    (-30.0, 30.0),   # alpha0p
    (-30.0, 30.0),   # alpha2p
]

# Run deterministic chi^2 minimization.
res = minimize(
    chi2_func,
    x0,
    method="L-BFGS-B",
    bounds=bounds,
    options={
        "maxiter": 300,
        "ftol": 1e-12,
        "gtol": 1e-8,
    },
)

print("\n--- Optimization result ---")
print("success :", res.success)
print("message :", res.message)
print("nfev    :", res.nfev)
print("nit     :", res.nit)
print("fun     :", res.fun)

print("\n--- Best-fit parameters from optimizer ---")
for name, val in zip(param_names, res.x):
    print(f"{name:10s} = {val:.10f}")

# Build the final best-fit parameter dictionary.
bestfit_params = dict(zip(param_names, res.x))
bestfit_params["b3p"] = 0.0
bestfit_params["alpha4p"] = 0.0
bestfit_params["alpha6p"] = 0.0

# Re-evaluate the likelihood at the optimized best-fit point.
try:
    likelihood(**bestfit_params)
    if hasattr(likelihood, "chi2"):
        chi2_bestfit = float(likelihood.chi2)
    else:
        chi2_bestfit = float(-2.0 * likelihood.loglikelihood)
except Exception as e:
    print("\nFinal likelihood evaluation failed:", repr(e))
    chi2_bestfit = np.nan

# Extract optimized cosmological parameters.
h_best = bestfit_params["h"]
omega_cdm_best = bestfit_params["omega_cdm"]
omega_b_best = bestfit_params["omega_b"]
n_s_best = bestfit_params["n_s"]
logA_best = bestfit_params["logA"]

# Compute derived parameters at the optimized best-fit point.
Omega_m_best = (omega_cdm_best + omega_b_best) / (h_best ** 2)
A_s_best = 1e-10 * np.exp(logA_best)

cosmo_best = Class()
sigma8_best = np.nan

try:
    cosmo_best.set({
        "output": "mPk",
        "Omega_cdm": omega_cdm_best / (h_best ** 2),
        "Omega_b": omega_b_best / (h_best ** 2),
        "h": h_best,
        "n_s": n_s_best,
        "A_s": A_s_best,
        "z_pk": 0.,
        "P_k_max_h/Mpc": 0.5,
    })
    cosmo_best.compute()
    sigma8_best = cosmo_best.sigma8()
    cosmo_best.struct_cleanup()
except Exception as e:
    print("\nCLASS evaluation for optimized sigma8 failed:", repr(e))

print("\n--- Optimized derived parameters ---")
print(f"chi^2    = {chi2_bestfit:.8f}")
print(f"sigma_8  = {sigma8_best:.8f}")
print(f"Omega_m  = {Omega_m_best:.8f}")

# Compare the best MCMC point with the direct optimizer result.
print("\n--- Comparison with MCMC best row ---")
print(f"MCMC best chi^2     = {best_row['chi2']:.8f}")
print(f"Optimizer best chi^2= {chi2_bestfit:.8f}")
print(f"Delta chi^2         = {chi2_bestfit - best_row['chi2']:.8e}")
print(f"MCMC best sigma8    = {best_row['sigma8']:.8f}")
print(f"Optimizer sigma8    = {sigma8_best:.8f}")
print(f"MCMC best Omega_m   = {best_row['Omega_m']:.8f}")
print(f"Optimizer Omega_m   = {Omega_m_best:.8f}")