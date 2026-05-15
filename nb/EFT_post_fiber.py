# ============================================================
# Full-shape correlation-function analysis with fibre-assignment systematics
# ============================================================
#
# Purpose:
#   This script constructs a mock two-point correlation function, applies the
#   fibre-assignment-induced distortion measured from pre/post catalogues, and
#   then fits the modified mock data with a full-shape LPT/Velocileptors model.
#
# Main workflow:
#   1. Load pre- and post-fibre-assignment correlation functions.
#   2. Compute the post/pre ratio for xi_0 and xi_2.
#   3. Build a fiducial cosmology with sigma8 ~= 0.8 using CLASS.
#   4. Generate theoretical mock correlation-function multipoles with desilike.
#   5. Apply the post/pre ratio to the mock multipoles.
#   6. Define a Gaussian likelihood using the modified mock and covariance matrix.
#   7. Build a Taylor emulator for faster likelihood evaluation.
#   8. Run MCMC sampling with ZeusSampler.
#   9. Compute derived parameters such as sigma8 and Omega_m.
#  10. Refine the best-fit point by direct chi^2 minimization.

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

# Duplicate imports from the original script are kept for minimal behavioral change.
# They can be safely removed in a cleaned production version.
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utility function: load stacked xi_0 and xi_2 data
# ============================================================

def load_and_split_data(filename):
    """
    Load a two-column correlation-function file and split it into multipoles.

    Expected input format:
        column 0: separation s
        column 1: correlation function xi

    The file is assumed to contain stacked multipoles. For example:
        s values for xi_0 increase from low to high,
        then s resets to a lower value,
        then s values for xi_2 increase from low to high.

    The reset in s is used to identify the boundary between xi_0 and xi_2.

    Parameters
    ----------
    filename : str
        Path to the input file.

    Returns
    -------
    s0 : ndarray
        Separation bins for the monopole.
    xi0 : ndarray
        Monopole correlation function.
    xi2 : ndarray or None
        Quadrupole correlation function. If no reset is found, returns None.
    """
    data = np.loadtxt(filename)
    s = data[:, 0]
    xi = data[:, 1]

    # A negative difference means that the s-coordinate has reset.
    # This is interpreted as the start of the next multipole block.
    split_indices = np.where(np.diff(s) < 0)[0] + 1

    if len(split_indices) > 0:
        split_idx = split_indices[0]

        # First block: monopole xi_0.
        s0 = s[:split_idx]
        xi0 = xi[:split_idx]

        # Second block: quadrupole xi_2.
        s2 = s[split_idx:]
        xi2 = xi[split_idx:]

        # Only s0 is returned because later code assumes the same s-grid for xi_0 and xi_2.
        return s0, xi0, xi2
    else:
        # Fallback for files containing only one multipole.
        return s, xi, None


# ============================================================
# Load pre/post fibre-assignment correlation functions
# ============================================================

# xi0_xi2_pre.dat : correlation function before fibre assignment.
# xi0_xi2_post.dat: correlation function after fibre assignment.
s_pre, xi0_pre, xi2_pre = load_and_split_data('xi0_xi2_pre.dat')
s_post, xi0_post, xi2_post = load_and_split_data('xi0_xi2_post.dat')

# Compute the multiplicative distortion induced by fibre assignment.
# This ratio will later be applied to a theoretical mock correlation function.
ratio_xi0 = xi0_post / xi0_pre

# Compute the quadrupole ratio only if xi_2 exists in both files.
if xi2_pre is not None and xi2_post is not None:
    ratio_xi2 = xi2_post / xi2_pre
else:
    ratio_xi2 = None



# Effective redshift of the sample.
z = 0.15

# Separation-bin edges for the correlation function.
# The analysis range is 40--160 Mpc/h with 20 bins.
sedges = np.linspace(40., 160., 21)

# Bin centers used for theoretical predictions and likelihood evaluation.
s = (sedges[:-1] + sedges[1:]) / 2.
ns = s.size

# k-bin edges used only to estimate the number of Fourier modes.
# These are passed to CorrelationFunctionStatistics for bookkeeping.
kedges = np.linspace(0., 0.3, 41)
nmodes = 4. * np.pi / 3. * (kedges[1:]**3 - kedges[:-1]**3)



# Fiducial cosmology. The target amplitude is sigma8 = 0.8.
h = 0.71
omega_b = 0.02258
Omega_m = 0.2648
n_s = 0.963
sigma8 = 0.8

# Convert total matter density Omega_m into physical CDM density omega_cdm.
# Here omega_cdm = Omega_cdm h^2 = Omega_m h^2 - omega_b.
omega_cdm = Omega_m * h**2 - omega_b


# ============================================================
# Find A_s corresponding to target sigma8 using CLASS
# ============================================================

# CLASS uses A_s as the primordial scalar amplitude, while the analysis often
# interprets results in terms of sigma8. This block numerically constructs an
# approximate mapping from sigma8 to A_s.
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

    # Compute sigma8 for each sampled A_s value.
    cosmo.set(params)
    cosmo.compute()
    sigma8_samples.append(cosmo.sigma8())

    # Clean CLASS internal structures before the next computation.
    cosmo.struct_cleanup()

As_array = np.array(As_samples)
s8_array = np.array(sigma8_samples)

# Fit A_s as a quadratic function of sigma8.
# This is used to interpolate the A_s value that gives sigma8 = 0.8.
coeffs = np.polyfit(s8_array, As_array, 2)
poly_func = np.poly1d(coeffs)

target_s8 = 0.8
target_As = poly_func(target_s8)
logA_target = np.log(1e10 * target_As)

# Create the fiducial cosmology object used by desilike/cosmoprimo.
fiducial_cosmo = Cosmology(
    h=h,
    omega_b=omega_b,
    n_s=n_s,
    omega_cdm=omega_cdm,
    A_s=target_As,
    Omega_k=0.0,
    engine='class',
)

# desilike commonly uses logA = ln(10^10 A_s).
logA_val = np.log(10**10 * fiducial_cosmo.A_s)
print(f"Calculated logA = {logA_val}")


# ============================================================
# Generate a theoretical mock correlation function
# ============================================================

# DirectPowerSpectrumTemplate provides the linear power-spectrum template.
template = DirectPowerSpectrumTemplate(z=z, fiducial=fiducial_cosmo)

# Set the fiducial cosmological parameters explicitly.
template.params.update({
    'h': h,
    'omega_b': omega_b,
    'omega_cdm': omega_cdm,
    'n_s': n_s,
    'logA': logA_val,
})

# Build the LPT/Velocileptors correlation-function model for ell = 0 and ell = 2.
theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(
    template=template,
    s=s,
    ells=(0, 2),
)

# Parameters used to generate the mock data.
# These include Lagrangian bias parameters and EFT counterterms.
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

# Evaluate the theoretical correlation-function multipoles.
theory(**mock_params)
print("LPT calculation computed.")

# theory.corr contains xi_0 and xi_2 evaluated at the s-bin centers.
poles = theory.corr

# Shot noise corresponding to an assumed number density nbar = 5e-4.
shotnoise = 1 / 5e-4

# Wrap the theoretical multipoles into a pypower statistics object.
# This object can be passed directly to the desilike observable.
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
print("xi0:", mean.corr[0][:5])
print("xi2:", mean.corr[1][:5])



# Covariance matrix for the post-fibre-assignment correlation function.
# It should match the data vector ordering and length used by the observable.
cov_path = 'cov_s_post.npy'
try:
    cov = np.load(cov_path)
    print(f"\nCovariance matrix loaded from {cov_path}")
except FileNotFoundError:
    print(f"\nWarning: Covariance file not found at {cov_path}. 'cov' variable is not set.")
    cov = None


# ============================================================
# Load post/pre ratios and apply them to the theoretical mock
# ============================================================

def load_ratios(pre_file, post_file):
    """
    Load pre/post correlation functions and compute post/pre ratios.

    The input files are assumed to have the same stacked multipole format:
        [xi_0 block, xi_2 block]

    The output ratios are used as multiplicative templates for the
    fibre-assignment distortion.

    Parameters
    ----------
    pre_file : str
        File containing the pre-fibre-assignment correlation function.
    post_file : str
        File containing the post-fibre-assignment correlation function.

    Returns
    -------
    s0 : ndarray
        Separation bins for xi_0.
    ratio0 : ndarray
        Post/pre ratio for xi_0.
    s2 : ndarray or None
        Separation bins for xi_2.
    ratio2 : ndarray or None
        Post/pre ratio for xi_2.
    """
    pre_data = np.loadtxt(pre_file)
    post_data = np.loadtxt(post_file)

    s_in = pre_data[:, 0]
    xi_pre = pre_data[:, 1]
    xi_post = post_data[:, 1]

    # Identify where the s-coordinate resets, i.e. the boundary between multipoles.
    split_indices = np.where(np.diff(s_in) < 0)[0] + 1

    if len(split_indices) > 0:
        split_idx = split_indices[0]

        # Monopole ratio.
        s0 = s_in[:split_idx]
        ratio0 = xi_post[:split_idx] / xi_pre[:split_idx]

        # Quadrupole ratio.
        s2 = s_in[split_idx:]
        ratio2 = xi_post[split_idx:] / xi_pre[split_idx:]

        return s0, ratio0, s2, ratio2
    else:
        # Fallback for files containing only one multipole.
        return s_in, xi_post / xi_pre, None, None


ratio_s0, ratio0, ratio_s2, ratio2 = load_ratios('xi0_xi2_pre.dat', 'xi0_xi2_post.dat')

# Build interpolation functions so that the ratio can be evaluated at the mock s-bins.
interp_ratio0 = interp1d(ratio_s0, ratio0, kind='linear', fill_value="extrapolate")
interp_ratio2 = interp1d(ratio_s2, ratio2, kind='linear', fill_value="extrapolate")

# Extract the theoretical mock multipoles.
s_mock = mean.s
xi0_mock = mean.corr[0]  # Monopole xi_0.
xi2_mock = mean.corr[1]  # Quadrupole xi_2.

# Evaluate the fibre-assignment ratios on the mock separation grid.
r0_interp = interp_ratio0(s_mock)
r2_interp = interp_ratio2(s_mock)

# Apply the fibre-assignment distortion multiplicatively.
# This produces the final mock data used in the likelihood.
xi0_final = xi0_mock * r0_interp
xi2_final = xi2_mock * r2_interp

# Replace the original mock multipoles with the distorted multipoles.
new_poles = mean.corr.copy()
new_poles[0] = xi0_final
new_poles[1] = xi2_final

# Rebuild the pypower statistics object with the modified multipoles.
# This is safer than modifying mean.corr in place because object mutability may
# depend on the installed pypower/desilike version.
mean_modified = CorrelationFunctionStatistics(
    mean.edges,
    mean.s,
    new_poles,
    nmodes=mean.nmodes,
    ells=mean.ells,
    shotnoise_nonorm=mean.shotnoise_nonorm,
    statistic='multipole',
)

# This object is the final data vector for the full-shape fit.
data = mean_modified

print("Ratio applied to mock data.")


# ============================================================
# Define the fitting model and parameter priors
# ============================================================

z = 0.15
template = DirectPowerSpectrumTemplate(z=z, fiducial=fiducial_cosmo)

# Fix selected template-level cosmological parameters.
# Note: later parts of the script still export or pass omega_b and n_s.
# Check desilike's parameter table if strict fixed/free behavior is important.
for param in ['omega_b', 'n_s']:
    template.params[param].update(fixed=True)

# The theory calculator used by the likelihood.
theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(template=template)

# Set priors, reference values, proposal widths, and fixed parameters.
# Free nuisance parameters include b1p, b2p, bsp, alpha0p, and alpha2p.
theory.params.update({
    'b1p': {
        'prior': {'dist': 'uniform', 'limits': [-0.375, 2.75]},
        'ref': 1.1327562,
        'proposal': 0.05,
    },

    'b2p': {
        'prior': {'dist': 'norm', 'loc': 0.0, 'scale': np.sqrt(10)},
        'ref': -0.34870614,
        'proposal': 0.5,
    },

    'bsp': {
        'prior': {'dist': 'norm', 'loc': 0.0, 'scale': np.sqrt(5)},
        'ref': -0.25395155,
        'proposal': 0.5,
    },

    'b3p': {'value': 0.0, 'fixed': True},

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

    'alpha4p': {'value': 0.0, 'fixed': True},
    'alpha6p': {'value': 0.0, 'fixed': True},
})


# ============================================================
# Build observable and Gaussian likelihood
# ============================================================

# The observable compares the modified mock data vector with the theory model.
# slim selects the separation range used in the fit for each multipole.
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

# Initial likelihood call to initialize internal quantities.
likelihood()


# ============================================================
# Build and save a Taylor emulator
# ============================================================

# The LPT/Velocileptors theory calculation can be expensive inside MCMC.
# A Taylor emulator is built around the theory calculator to accelerate sampling.
emulator = Emulator(
    theory,
    engine=TaylorEmulatorEngine(order={'*': 2, 'sn0': 1}),
)

# Generate samples for the emulator and fit the Taylor expansion.
emulator.set_samples()
emulator.fit()

# Save the emulator for later reuse.
base_dir = '_tests/EFT_post_check'
EFT_pre_emulator_fn_cosmo = os.path.join(base_dir, 'emulator_4.npy')
emulator.save(EFT_pre_emulator_fn_cosmo)

# Replace the exact theory calculator in the observable with the emulator calculator.
observable.init.update(theory=emulator.to_calculator())


# ============================================================
# Run MCMC sampling
# ============================================================

# ZeusSampler samples the posterior defined by the likelihood and parameter priors.
sampler = ZeusSampler(
    likelihood,
    save_fn='_tests/EFT_post_check/chain_fs_direct_4_*.npy',
    seed=42,
)

# Run until the convergence criterion is satisfied.
sampler.run(check={'max_eigen_gr': 0.1})


# ============================================================
# Diagnostic check at the fiducial/mock parameter point
# ============================================================

# Evaluate the likelihood at the parameter values used to generate the mock.
# This helps diagnose how much the fibre-assignment ratio shifts the data away
# from the unmodified fiducial model.
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

    # If available, compare the flattened data vector and theory vector directly.
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
# 14. Remove burn-in and summarize the MCMC chain
# ============================================================

# Remove the first 50% of samples as burn-in.
chain = sampler.chains[0].remove_burnin(0.5)
print(chain.to_stats(tablefmt='pretty'))


# ============================================================
# Compute sigma8 for every MCMC sample using CLASS
# ============================================================

# The chain samples logA, but sigma8 is a more interpretable derived parameter.
# Therefore, CLASS is run at each chain point to compute sigma8.
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
# Compute Omega_m and export the chain to CSV
# ============================================================

# Derived matter density parameter:
#   Omega_m = (omega_cdm + omega_b) / h^2
Omega_m_chain = (chain['omega_cdm'] + chain['omega_b']) / (chain['h']**2)

print("Preparing data for CSV export, including chi^2 and Omega_m...")

# Flatten walker and step dimensions for CSV output.
h_flat = chain['h'].flatten()
omega_cdm_flat = chain['omega_cdm'].flatten()
omega_b_flat = chain['omega_b'].flatten()

omegam_flat = (omega_cdm_flat + omega_b_flat) / (h_flat**2)
sigma8_flat = sigma8_chain.flatten()

# Convert log-likelihood into chi^2 using chi^2 = -2 log L.
logl_flat = chain['loglikelihood'].flatten()
chi2_flat = -2.0 * logl_flat

# Collect sampled and derived parameters.
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

# Save the full chain table for later plotting and analysis.
df = pd.DataFrame(data_dict)
output_csv = "mcmc_chain_results_with_chi2_minimize_post_1.csv"
df.to_csv(output_csv, index=False)

print(f"Success. Chain data, Omega_m, and chi^2 saved to {output_csv}. Total samples: {len(df)}")


# ============================================================
# Direct minimization of chi^2 using the likelihood
# ============================================================

print("\n================================================")
print("Starting direct chi^2 minimization using likelihood")
print("================================================")

# Use the best MCMC sample as the starting point for deterministic optimization.
# This makes the optimizer start in a high-posterior / low-chi^2 region.
best_idx = df["chi2"].idxmin()
best_row = df.loc[best_idx]

print("\n--- MCMC best point used as optimizer start ---")
print(f"Row Index    : {best_idx}")
print(f"chi^2        : {best_row['chi2']:.8f}")
print(f"sigma_8      : {best_row['sigma8']:.8f}")
print(f"Omega_m      : {best_row['Omega_m']:.8f}")

# Parameters varied by the direct optimizer.
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

x0 = np.array([best_row[p] for p in param_names], dtype=float)

# Counters used to monitor optimization progress.
eval_counter = {"n": 0}
best_seen = {"chi2": np.inf, "x": None}


def chi2_func(x):
    """
    Objective function for scipy.optimize.minimize.

    Parameters
    ----------
    x : ndarray
        Array of parameter values ordered according to param_names.

    Returns
    -------
    chi2_val : float
        The chi^2 value evaluated by the desilike likelihood.
    """
    eval_counter["n"] += 1

    # Convert the optimizer vector into a parameter dictionary.
    params = dict(zip(param_names, x))

    # Fixed parameters not included in x.
    params["b3p"] = 0.0
    params["alpha4p"] = 0.0
    params["alpha6p"] = 0.0

    try:
        likelihood(**params)

        if hasattr(likelihood, "chi2"):
            chi2_val = float(likelihood.chi2)
        else:
            chi2_val = float(-2.0 * likelihood.loglikelihood)

        # Protect the optimizer from invalid likelihood evaluations.
        if not np.isfinite(chi2_val):
            return 1e30

        # Print whenever a new best chi^2 is found.
        if chi2_val < best_seen["chi2"]:
            best_seen["chi2"] = chi2_val
            best_seen["x"] = np.array(x, copy=True)
            print(f"[eval {eval_counter['n']:5d}] new best chi2 = {chi2_val:.8f}")

        return chi2_val

    except Exception as e:
        # Failed evaluations are assigned a very large chi^2.
        print(f"[eval {eval_counter['n']:5d}] FAILED: {repr(e)}")
        return 1e30


# Check the chi^2 value at the MCMC-best starting point.
chi2_x0 = chi2_func(x0)
print(f"\nchi2(x0) = {chi2_x0:.8f}")

# Bounds used by L-BFGS-B.
# These ranges are intentionally broad but physically reasonable.
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

# Run bound-constrained minimization.
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

# Evaluate the likelihood once more at the optimizer best-fit point.
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

# Compare the best MCMC sample with the deterministic optimizer result.
print("\n--- Comparison with MCMC best row ---")
print(f"MCMC best chi^2     = {best_row['chi2']:.8f}")
print(f"Optimizer best chi^2= {chi2_bestfit:.8f}")
print(f"Delta chi^2         = {chi2_bestfit - best_row['chi2']:.8e}")
print(f"MCMC best sigma8    = {best_row['sigma8']:.8f}")
print(f"Optimizer sigma8    = {sigma8_best:.8f}")
print(f"MCMC best Omega_m   = {best_row['Omega_m']:.8f}")
print(f"Optimizer Omega_m   = {Omega_m_best:.8f}")
