# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 2 — Core Transformations I: JIT Compilation & Automatic Differentiation
#
# **Estimated time:** ~25 minutes
#
# In this notebook, we'll explore two of JAX's most powerful transformations:
#
# 1. **`jax.jit`** — Just-In-Time compilation for dramatic speed improvements
# 2. **`jax.grad`** — Automatic differentiation for computing exact gradients
#
# These two transformations form the foundation of JAX's power for scientific computing. By the end of this notebook, you'll understand how to accelerate your code with JIT and compute gradients effortlessly — essential skills for optimization, inference, and sensitivity analysis in astrophysics.

# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

# Check JAX version and available devices
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")


# %% [markdown]
# ---
#
# ## 2.1 Just-In-Time Compilation with `jax.jit`
#
# ### What is JIT compilation?
#
# When you write a Python function using JAX operations, it normally executes line-by-line in Python's interpreter. This is flexible but slow. **JIT compilation** transforms your function into optimized machine code using Google's XLA (Accelerated Linear Algebra) compiler.
#
# The process works in two stages:
# 1. **Tracing:** JAX runs your function with abstract "tracer" values to build a computation graph (Directed Acyclic Graph, DAG)
# 2. **Compilation:** XLA compiles this graph into highly optimized code for your hardware (CPU/GPU/TPU)
#
# ```
#       Input a ───────┐
#                      ▼
#                   ┌─────────┐      Intermediate c
#       Input b ───▶│  * (mul)│───────────┐
#                │  └─────────┘           │
#                │                        ▼
#                │                 ┌───────────┐      Output d
#                └────────────────▶│  + (add)  │────────▶
#                                  └───────────┘
# ```
# a/b: source nodes (tensors/arrays); Operator mul/add: computational nodes; edge:how data flows between nodes
#
# The first call is slower (compilation overhead), but subsequent calls are much faster.
#
# ### Basic usage
#
# Let's start with a simple astrophysics example: computing the luminosity distance in a flat ΛCDM cosmology.

# %%
def luminosity_distance(z, H0=70.0, Omega_m=0.3):
    """
    Compute luminosity distance in Mpc for redshift z in a flat ΛCDM cosmology.
    
    Parameters:
    -----------
    z : float
        Redshift
    H0 : float
        Hubble constant in km/s/Mpc
    Omega_m : float
        Matter density parameter
    
    Returns:
    --------
    d_L : float
        Luminosity distance in Mpc
    """
    c = 299792.458  # speed of light in km/s
    Omega_lambda = 1.0 - Omega_m
    
    # Numerical integration of comoving distance
    z_array = jnp.linspace(0, z, 1000)
    E_z = jnp.sqrt(Omega_m * (1 + z_array)**3 + Omega_lambda)
    integrand = 1.0 / E_z
    d_c = jnp.trapezoid(integrand, z_array) * (c / H0)
    
    # Luminosity distance
    d_L = (1 + z) * d_c
    return d_L

# Test the function
z_test = 1.0
d_L = luminosity_distance(z_test)
print(f"Luminosity distance at z={z_test}: {d_L:.2f} Mpc")

# %% [markdown]
# ### Applying JIT compilation
#
# Now let's compile this function with `jax.jit`. We can do this in two ways:

# %%
# Method 1: Wrap the function
luminosity_distance_jit = jax.jit(luminosity_distance)

# Method 2: Use as a decorator (more common)
@jax.jit
def luminosity_distance_jit_v2(z, H0=70.0, Omega_m=0.3):
    c = 299792.458
    Omega_lambda = 1.0 - Omega_m
    z_array = jnp.linspace(0, z, 1000)
    E_z = jnp.sqrt(Omega_m * (1 + z_array)**3 + Omega_lambda)
    integrand = 1.0 / E_z
    d_c = jnp.trapezoid(integrand, z_array) * (c / H0)
    d_L = (1 + z) * d_c
    return d_L

# Verify they give the same result
print(f"Non-JIT: {luminosity_distance(1.0):.2f} Mpc")
print(f"JIT (method 1): {luminosity_distance_jit(1.0):.2f} Mpc")
print(f"JIT (method 2): {luminosity_distance_jit_v2(1.0):.2f} Mpc")


# %% [markdown]
# ### Benchmarking: JIT vs non-JIT
#
# Let's measure the performance difference. We'll time both the first call (which includes compilation) and subsequent calls.

# %%
# Create a fresh JIT-compiled function for timing
@jax.jit
def luminosity_distance_jit_timed(z, H0=70.0, Omega_m=0.3):
    c = 299792.458
    Omega_lambda = 1.0 - Omega_m
    z_array = jnp.linspace(0, z, 1000)
    E_z = jnp.sqrt(Omega_m * (1 + z_array)**3 + Omega_lambda)
    integrand = 1.0 / E_z
    d_c = jnp.trapezoid(integrand, z_array) * (c / H0)
    d_L = (1 + z) * d_c
    return d_L

# Timing non-JIT version
z_test = 1.5
n_runs = 100

start = time.time()
for _ in range(n_runs):
    result = luminosity_distance(z_test)
    result.block_until_ready()  # Wait for computation to finish (important for GPU timing)
end = time.time()
time_no_jit = (end - start) / n_runs

# Timing JIT version - first call (includes compilation)
start = time.time()
result = luminosity_distance_jit_timed(z_test)
result.block_until_ready()
end = time.time()
time_jit_first = end - start

# Timing JIT version - subsequent calls
start = time.time()
for _ in range(n_runs):
    result = luminosity_distance_jit_timed(z_test)
    result.block_until_ready()
end = time.time()
time_jit_subsequent = (end - start) / n_runs

print(f"Non-JIT average time: {time_no_jit*1000:.3f} ms")
print(f"JIT first call (with compilation): {time_jit_first*1000:.3f} ms")
print(f"JIT subsequent calls: {time_jit_subsequent*1000:.3f} ms")
print(f"\
Speedup: {time_no_jit/time_jit_subsequent:.1f}x faster")


# %% [markdown]
# ### Understanding tracing with abstract values
#
# When JAX traces your function, it doesn't use concrete values like `1.5` or `2.0`. Instead, it uses **abstract values** (tracers) that represent the shape and dtype of your data. This allows JAX to build a computation graph that works for any input with the same shape/dtype.
#
# Let's see what happens during tracing:

# %%
@jax.jit
def simple_function(x):
    print(f"Tracing with x = {x}")  # This prints during tracing
    print(f"Type of x: {type(x)}")
    y = x ** 2 + 2 * x + 1
    return y

print("First call:")
result1 = simple_function(3.0)
print(f"Result: {result1}\
")

print("Second call (no recompilation):")
result2 = simple_function(5.0)
print(f"Result: {result2}\
")

print("Third call with different shape (triggers recompilation):")
result3 = simple_function(jnp.array([1.0, 2.0, 3.0]))
print(f"Result: {result3}")


# %% [markdown]
# **Key observation:** The print statements only appear during tracing (first call and when shape changes). The compiled function doesn't include Python print statements — they're only executed during the tracing phase.
#
# ### Common pitfall: Data-dependent control flow
#
# One of the most common JIT errors occurs when you use data-dependent control flow (like `if x > 0`) inside a JIT-compiled function. Let's see why this fails:

# %%
# This will FAIL when JIT-compiled
def square_if_positive_bad(x):
    """Square the input if it's positive, otherwise return 0."""
    if x > 0:  # Data-dependent control flow!
        return x ** 2
    else:
        return 0.0

# Works fine without JIT
print("Without JIT:")
print(f"x=5: {square_if_positive_bad(5.0)}")
print(f"x=-1: {square_if_positive_bad(-1.0)}")

# But fails with JIT
print("With JIT:")
try:
    square_if_positive_bad_jit = jax.jit(square_if_positive_bad)
    result = square_if_positive_bad_jit(5.0)
    print(f"With JIT: {result}")
except Exception as e:
    print(f"JIT Error: {type(e).__name__}")
    print(f"Message: {str(e)[:200]}...")  # Print first 200 chars


# %% [markdown]
# **Why does this fail?** During tracing, `x` is an abstract tracer, not a concrete number. JAX can't evaluate `x > 0` to decide which branch to take, because it doesn't know the actual value yet.
#
# **Solution:** Use `jax.lax.cond` for conditional logic inside JIT-compiled functions, or use `jnp.where` for simple cases:
#
# The above structured control flow operators provided by JAX defer conditional evaluation until actual execution on the GPU/CPU, rather than during the tracing stage.

# %%
# Solution 1: Use jnp.where (simpler for this case)
# jnp.where is best for simple, element-wise choices.
# Note: It evaluates BOTH branches and then selects the result.
@jax.jit
def square_if_positive_good(x):
    """Square the input if it's positive, otherwise return 0."""
    return jnp.where(x > 0, x ** 2, 0.0)

# Solution 2: Use jax.lax.cond (more general)
# jax.lax.cond acts like a true if-else statement.
# It only executes the branch that matches the condition at runtime.
@jax.jit
def square_if_positive_good_v2(x):
    """Square the input if it's positive, otherwise return 0."""
    return jax.lax.cond(
        x > 0,
        lambda val: val ** 2,  # true branch
        lambda val: 0.0,       # false branch
        x
    )

print("With JIT (using jnp.where):")
print(f"x=5: {square_if_positive_good(5.0)}")
print(f"x=-1: {square_if_positive_good(-1.0)}")

print("\nWith JIT (using jax.lax.cond):")
print(f"x=5: {square_if_positive_good_v2(5.0)}")
print(f"x=-1: {square_if_positive_good_v2(-1.0)}")


# %% [markdown]
# ### `static_argnums` and `static_argnames`
#
# Sometimes you want certain arguments to be treated as **static** — meaning they're not traced, and changing them triggers recompilation. This is useful for things like array sizes, model configurations, or flags.
#
# Example: Computing a power-law spectrum with a variable number of frequency bins:

# %%
# Without static_argnums, this would fail because n_bins is used to create an array
# @functools.partial(jit, static_argnums=(3,)) or
@jax.jit(static_argnums=(3,))  # Mark n_bins (4th argument, index 3) as static
def power_law_spectrum(amplitude, index, nu_ref, n_bins=100):
    """
    Compute a power-law spectrum: F(nu) = amplitude * (nu / nu_ref)^index
    """
    print(f"--- Compiling power_law_spectrum with n_bins={n_bins} ---")
    nu = jnp.logspace(jnp.log10(nu_ref) - 1, jnp.log10(nu_ref) + 1, n_bins)
    flux = amplitude * (nu / nu_ref)**index
    return nu, flux

# Test with different n_bins values
print("\nFirst call (n_bins=50):")
nu1, flux1 = power_law_spectrum(1.0, -0.7, 1e9, n_bins=50)

print("\nSecond call (n_bins=50, SAME value, should NOT recompile):")
nu1_repeat, flux1_repeat = power_law_spectrum(1.0, -0.7, 1e9, n_bins=50)

print("\nThird call (n_bins=200, DIFFERENT value, SHOULD recompile):")
nu2, flux2 = power_law_spectrum(1.0, -0.7, 1e9, n_bins=200)

print(f"\nResults: {len(nu1)} vs {len(nu2)} frequency bins")


# %% [markdown]
# ---
#
# ## 2.2 Automatic Differentiation with `jax.grad`
#
# ### The power of autodiff
#
# Automatic differentiation (autodiff) is one of JAX's killer features. It computes **exact** gradients efficiently, without numerical approximation or manual derivation. This is essential for:
#
# - **Optimization:** Gradient descent, L-BFGS, etc.
# - **Bayesian inference:** Hamiltonian Monte Carlo (HMC), NUTS
# - **Sensitivity analysis:** Understanding how outputs depend on inputs
# - **Inverse problems:** Fitting models to data
#
# JAX uses **reverse-mode autodiff** (backpropagation) by default, which is efficient for functions with many inputs and few outputs (like loss functions).
#
# ### Basic usage: `jax.grad`
#
# Let's compute the gradient of a simple astrophysical function: the Schwarzschild radius as a function of mass.

# %%
def schwarzschild_radius(mass_solar):
    """
    Compute Schwarzschild radius in km for a given mass in solar masses.
    R_s = 2GM/c^2
    """
    G = 6.674e-11  # m^3 kg^-1 s^-2
    c = 2.998e8    # m/s
    M_sun = 1.989e30  # kg
    
    mass_kg = mass_solar * M_sun
    R_s = 2 * G * mass_kg / c**2  # in meters
    return R_s / 1000  # convert to km

# Compute the gradient: dR_s/dM
grad_schwarzschild = jax.grad(schwarzschild_radius)

# Test at 10 solar masses
M = 10.0
R_s = schwarzschild_radius(M)
dR_dM = grad_schwarzschild(M)

print(f"Mass: {M} M_sun")
print(f"Schwarzschild radius: {R_s:.2f} km")
print(f"Gradient dR_s/dM: {dR_dM:.3f} km/M_sun")

# Verify: For R_s = 2GM/c^2, we expect dR_s/dM = 2G/c^2 (constant)
G = 6.674e-11
c = 2.998e8
M_sun = 1.989e30
expected_gradient = 2 * G * M_sun / c**2 / 1000
print(f"\
Expected gradient (analytical): {expected_gradient:.3f} km/M_sun")
print(f"Match: {jnp.allclose(dR_dM, expected_gradient)}")


# %% [markdown]
# ### `argnums`: Choosing which argument to differentiate
#
# For functions with multiple arguments, use `argnums` to specify which argument(s) to differentiate with respect to:

# %%
def kepler_third_law(a, M_star):
    """
    Compute orbital period using Kepler's third law.
    P^2 = (4π^2/GM) * a^3
    
    Parameters:
    -----------
    a : float
        Semi-major axis in AU
    M_star : float
        Stellar mass in solar masses
    
    Returns:
    --------
    P : float
        Orbital period in years
    """
    G = 4 * jnp.pi**2  # in AU^3 / (M_sun * year^2)
    P_squared = (4 * jnp.pi**2 / (G * M_star)) * a**3
    return jnp.sqrt(P_squared)

# Gradient with respect to semi-major axis (default: argnums=0)
grad_wrt_a = jax.grad(kepler_third_law, argnums=0)

# Gradient with respect to stellar mass
grad_wrt_M = jax.grad(kepler_third_law, argnums=1)

# Gradients with respect to both (returns a tuple)
grad_wrt_both = jax.grad(kepler_third_law, argnums=(0, 1))

# Test case: Earth-like orbit around Sun-like star
a = 1.0  # AU
M = 1.0  # M_sun

P = kepler_third_law(a, M)
dP_da = grad_wrt_a(a, M)
dP_dM = grad_wrt_M(a, M)
dP_da_both, dP_dM_both = grad_wrt_both(a, M)

print(f"Orbital period: {P:.3f} years")
print(f"\
Gradients:")
print(f"  dP/da = {dP_da:.3f} years/AU")
print(f"  dP/dM = {dP_dM:.3f} years/M_sun")
print(f"\
Using argnums=(0,1): dP/da = {dP_da_both:.3f}, dP/dM = {dP_dM_both:.3f}")


# %% [markdown]
# ### `jax.value_and_grad`: Get both value and gradient
#
# Often you need both the function value and its gradient (e.g., in optimization). Instead of calling the function twice, use `jax.value_and_grad`:

# %%
def chi_squared(model_params, data, sigma):
    """
    Compute chi-squared for a simple linear model: y = a*x + b
    
    Parameters:
    -----------
    model_params : tuple (a, b)
        Model parameters
    data : tuple (x, y)
        Observed data
    sigma : float
        Measurement uncertainty
    """
    a, b = model_params
    x, y = data
    y_model = a * x + b
    chi2 = jnp.sum(((y - y_model) / sigma)**2)
    return chi2

# Create synthetic data
np.random.seed(42)
x_data = jnp.linspace(0, 10, 20)
sigma = 0.5
y_data = 2.5 * x_data + 1.0 + np.random.normal(0, sigma, size=20)

# Test parameters
params = (2.0, 0.5)

# Method 1: Separate calls (inefficient)
chi2_val = chi_squared(params, (x_data, y_data), sigma)
# By default, jax.grad computes the gradient with respect to the first argument (argnums=0).
# In this case, that's 'model_params'.
# jax.grad(chi_squared, argnums=0)
grad_chi2 = jax.grad(chi_squared)(params, (x_data, y_data), sigma)

print("Method 1 (separate calls):")
print(f"  χ² = {chi2_val:.3f}")
print(f"  ∇χ² = {grad_chi2}")

# Method 2: Combined call (efficient)
value_and_grad_chi2 = jax.value_and_grad(chi_squared)
chi2_val, grad_chi2 = value_and_grad_chi2(params, (x_data, y_data), sigma)

print("\
Method 2 (value_and_grad):")
print(f"  χ² = {chi2_val:.3f}")
print(f"  ∇χ² = {grad_chi2}")


# %% [markdown]
# ### Forward-mode vs reverse-mode differentiation
#
# JAX supports both **forward-mode** and **reverse-mode** autodiff:
#
# - **Reverse-mode** (`jax.grad`): Efficient for functions with many inputs, few outputs (e.g., loss functions). Complexity scales with output dimension.
# - **Forward-mode** (`jax.jvp`): Efficient for functions with few inputs, many outputs (e.g., Jacobian-vector products). Complexity scales with input dimension.
#
# For most machine learning and optimization tasks, reverse-mode is the right choice. Forward-mode is useful for computing directional derivatives or when you have very high-dimensional outputs.
#
# **Rule of thumb:** Use `jax.grad` (reverse-mode) unless you have a specific reason to use forward-mode.

# %% [markdown]
# ---
#
# ## 2.3 Higher-Order and Structured Derivatives
#
# ### `jax.jacobian` and `jax.hessian`
#
# For vector-valued functions, you often need the full Jacobian matrix (all first-order partial derivatives) or Hessian matrix (all second-order partial derivatives).

# %%
def cartesian_to_spherical(xyz):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Parameters:
    -----------
    xyz : array of shape (3,)
        Cartesian coordinates [x, y, z]
    
    Returns:
    --------
    rtp : array of shape (3,)
        Spherical coordinates [r, theta, phi]
    """
    x, y, z = xyz
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)  # polar angle
    phi = jnp.arctan2(y, x)    # azimuthal angle
    return jnp.array([r, theta, phi])

# Compute the Jacobian matrix
jacobian_fn = jax.jacobian(cartesian_to_spherical)

# Test point
xyz = jnp.array([1.0, 1.0, 1.0])
rtp = cartesian_to_spherical(xyz)
J = jacobian_fn(xyz)

# Note: jax.grad only works for SCALAR-valued functions.
# Since cartesian_to_spherical returns a vector of shape (3,),
# jax.grad(cartesian_to_spherical) would raise a TypeError.
# We must use jax.jacobian for vector-valued functions.
print("Attempting jax.grad on a vector function:")
try:
    jax.grad(cartesian_to_spherical)(xyz)
except TypeError as e:
    print(f"  Caught expected error: {e}")

print("\nCartesian coordinates:", xyz)
print("Spherical coordinates:", rtp)
print("\
Jacobian matrix (∂[r,θ,φ]/∂[x,y,z]):")
print(J)
print(f"\
Shape: {J.shape}")


# %%
# Hessian example: second derivatives of a scalar function
# return a vector of second derivatives (for scalar functions)
# or full Hessian matrix (for vector-valued functions)
def gravitational_potential(r, M=1.0):
    """
    Gravitational potential: Φ(r) = -GM/r
    (Using G=1 for simplicity)
    """
    return -M / r

# Compute Hessian (second derivative for 1D input)
hessian_fn = jax.hessian(gravitational_potential)

r_test = 2.0
phi = gravitational_potential(r_test)
grad_phi = jax.grad(gravitational_potential)(r_test)
hess_phi = hessian_fn(r_test)

print(f"At r = {r_test}:")
print(f"  Φ(r) = {phi:.4f}")
print(f"  dΦ/dr = {grad_phi:.4f}")
print(f"  d²Φ/dr² = {hess_phi:.4f}")

# Analytical: Φ = -M/r, dΦ/dr = M/r², d²Φ/dr² = -2M/r³
M = 1.0
analytical_grad = M / r_test**2
analytical_hess = -2 * M / r_test**3
print(f"\
Analytical values:")
print(f"  dΦ/dr = {analytical_grad:.4f}")
print(f"  d²Φ/dr² = {analytical_hess:.4f}")


# %% [markdown]
# ### Composing `grad` of `grad`: Higher-order derivatives
#
# You can compose gradient operations to get higher-order derivatives. This is trivial in JAX!

# %%
def f(x):
    """A simple polynomial: f(x) = x^4 - 3x^2 + 2x"""
    return x**4 - 3*x**2 + 2*x

# First derivative
df_dx = jax.grad(f)

# Second derivative (grad of grad)
d2f_dx2 = jax.grad(df_dx)

# Third derivative
d3f_dx3 = jax.grad(d2f_dx2)

# Fourth derivative
d4f_dx4 = jax.grad(d3f_dx3)

x = 2.0
print(f"At x = {x}:")
print(f"  f(x) = {f(x):.4f}")
print(f"  f'(x) = {df_dx(x):.4f}")
print(f"  f''(x) = {d2f_dx2(x):.4f}")
print(f"  f'''(x) = {d3f_dx3(x):.4f}")
print(f"  f''''(x) = {d4f_dx4(x):.4f}")

# Analytical: f(x) = x^4 - 3x^2 + 2x
# f'(x) = 4x^3 - 6x + 2
# f''(x) = 12x^2 - 6
# f'''(x) = 24x
# f''''(x) = 24
print(f"\
Analytical at x = {x}:")
print(f"  f'(x) = {4*x**3 - 6*x + 2:.4f}")
print(f"  f''(x) = {12*x**2 - 6:.4f}")
print(f"  f'''(x) = {24*x:.4f}")
print(f"  f''''(x) = {24:.4f}")

# %% [markdown]
# ### Brief note on `jax.custom_jvp` and `jax.custom_vjp`
#
# Sometimes you want to define custom gradient rules for numerical stability or efficiency. JAX provides `jax.custom_jvp` (custom forward-mode) and `jax.custom_vjp` (custom reverse-mode) for this purpose.
#
# **Example use case:** Implementing a numerically stable version of `log(1 + exp(x))` (log-sum-exp trick) with a custom gradient.
#
# We won't cover this in detail here, but it's worth knowing these tools exist for advanced use cases. See the [JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html) for more information.

# %% [markdown]
# ---
#
# ## 2.4 Hands-On Exercise: Fitting a Power-Law Model with Gradient Descent
#
# Now let's put everything together! We'll fit a power-law model to synthetic astrophysical data using gradient descent, leveraging both JIT compilation and automatic differentiation.
#
# ### The problem
#
# Many astrophysical phenomena follow power-law distributions:
# - X-ray spectra: $F(E) = A \cdot E^{-\Gamma}$
# - Galaxy luminosity functions
# - Stellar mass functions
# - Cosmic ray energy spectra
#
# We'll generate synthetic data from a power-law model and recover the parameters using gradient descent.

# %%
# Step 1: Generate synthetic data
# True model: F(E) = A * E^(-Gamma)
A_true = 10.0      # Normalization
Gamma_true = 2.0   # Spectral index

# Energy bins (in keV)
E = jnp.logspace(0, 2, 50)  # 1 to 100 keV

# True flux
F_true = A_true * E**(-Gamma_true)

# Add Poisson noise (realistic for photon counting)
key = jax.random.PRNGKey(42)
# Simulate counts (assuming some exposure time)
exposure = 1000.0
counts_expected = F_true * exposure
counts_observed = jax.random.poisson(key, counts_expected).astype(float)
F_observed = counts_observed / exposure

# Uncertainties (Poisson: sigma = sqrt(N))
sigma = jnp.sqrt(counts_observed) / exposure
# Avoid division by zero
sigma = jnp.where(sigma > 0, sigma, 1.0)

print(f"True parameters: A = {A_true}, Γ = {Gamma_true}")
print(f"Energy range: {E[0]:.1f} - {E[-1]:.1f} keV")
print(f"Number of data points: {len(E)}")


# %%
# Step 2: Define the model and loss function
def power_law_model(params, E):
    """Power-law model: F(E) = A * E^(-Gamma)"""
    A, Gamma = params
    return A * E**(-Gamma)

def chi_squared_loss(params, E, F_obs, sigma):
    """Chi-squared loss function"""
    F_model = power_law_model(params, E)
    residuals = (F_obs - F_model) / sigma
    chi2 = jnp.sum(residuals**2)
    return chi2

# Test the loss function
params_init = jnp.array([5.0, 1.5])  # Initial guess (deliberately wrong)
loss_init = chi_squared_loss(params_init, E, F_observed, sigma)
print(f"Initial parameters: A = {params_init[0]:.2f}, Γ = {params_init[1]:.2f}")
print(f"Initial χ² = {loss_init:.2f}")


# %%
# Step 3: Set up gradient descent with JIT compilation
@jax.jit
def gradient_descent_step(params, E, F_obs, sigma, learning_rate):
    """Single gradient descent step (JIT-compiled for speed)"""
    loss, grad = jax.value_and_grad(chi_squared_loss)(params, E, F_obs, sigma)
    params_new = params - learning_rate * grad
    return params_new, loss

# Run gradient descent
params = params_init.copy()
learning_rate = 1e-5
n_iterations = 5000

losses = []
params_history = [params.copy()]

for i in range(n_iterations):
    params, loss = gradient_descent_step(params, E, F_observed, sigma, learning_rate)
    losses.append(float(loss))
    params_history.append(params.copy())
    
    if i % 1000 == 0:
        print(f"Iteration {i:3d}: χ² = {loss:.2f}, A = {params[0]:.3f}, Γ = {params[1]:.3f}")

# Pro Tip: For production code, you would wrap this loop in jax.lax.scan to avoid Python overhead.
# But for a tutorial, this explicit loop is clearer.

print(f"\
Final parameters: A = {params[0]:.3f}, Γ = {params[1]:.3f}")
print(f"True parameters:  A = {A_true:.3f}, Γ = {Gamma_true:.3f}")
print(f"Final χ² = {losses[-1]:.2f}")

# %%
# Step 4: Visualize the results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Data and fit
ax = axes[0]
ax.errorbar(E, F_observed, yerr=sigma, fmt='o', alpha=0.6, label='Observed data', markersize=4)
ax.plot(E, F_true, 'k--', linewidth=2, label=f'True model (A={A_true}, Γ={Gamma_true})', alpha=1)
F_fit = power_law_model(params, E)
ax.plot(E, F_fit, 'r-', linewidth=2, label=f'Fitted model (A={params[0]:.2f}, Γ={params[1]:.2f})', alpha=0.5)
ax.set_xlabel('Energy (keV)', fontsize=12)
ax.set_ylabel('Flux (arbitrary units)', fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.set_title('Power-Law Spectrum Fit', fontsize=13)
ax.grid(True, alpha=0.3)

# Panel 2: Loss convergence
ax = axes[1]
ax.plot(losses, linewidth=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('χ² Loss', fontsize=12)
ax.set_title('Optimization Convergence', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Panel 3: Parameter trajectory
ax = axes[2]
params_history = jnp.array(params_history)
ax.plot(params_history[:, 0], params_history[:, 1], 'o-', markersize=1, alpha=0.6, linewidth=1)
ax.plot(params_init[0], params_init[1], 'go', markersize=10, label='Initial', zorder=5)
ax.plot(A_true, Gamma_true, 'k*', markersize=15, label='True', zorder=5)
ax.plot(params[0], params[1], 'ro', markersize=10, label='Final', zorder=5)
ax.set_xlabel('Amplitude A', fontsize=12)
ax.set_ylabel('Spectral Index Γ', fontsize=12)
ax.set_title('Parameter Space Trajectory', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\
✓ Successfully fitted a power-law model using JAX's jit and grad!")

# %% [markdown]
# ---
#
# ## Summary
#
# In this notebook, we covered two fundamental JAX transformations:
#
# ### `jax.jit` — Just-In-Time Compilation
# - Compiles Python functions to optimized machine code via XLA
# - First call includes compilation overhead, subsequent calls are much faster
# - Works by tracing functions with abstract values
# - **Pitfall:** Data-dependent control flow requires `jax.lax.cond` or `jnp.where`
# - Use `static_argnums` for arguments that shouldn't be traced
#
# ### `jax.grad` — Automatic Differentiation
# - Computes exact gradients efficiently using reverse-mode autodiff
# - Use `argnums` to specify which arguments to differentiate
# - `jax.value_and_grad` returns both function value and gradient
# - `jax.jacobian` and `jax.hessian` for structured derivatives
# - Gradients compose trivially: `grad(grad(f))` for second derivatives
#
# ### Key Takeaways
# 1. **JIT compilation** can provide 10-100x speedups for numerical code
# 2. **Automatic differentiation** eliminates manual gradient derivation
# 3. These transformations **compose**: you can JIT-compile gradient functions
# 4. Together, they enable efficient gradient-based optimization and inference
#
# In the next notebook, we'll explore `jax.vmap` for automatic vectorization and see how these three transformations (jit, grad, vmap) compose beautifully together.

# %%
