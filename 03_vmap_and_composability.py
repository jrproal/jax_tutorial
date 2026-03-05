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
# # Notebook 3 — Core Transformations II: Vectorization with `jax.vmap`
#
# **Estimated time:** 20 minutes
#
# In this notebook, we'll explore one of JAX's most powerful features: **automatic vectorization** with `jax.vmap`. We'll see how `vmap` eliminates the need for manual batching and broadcasting, and how it composes beautifully with `jit` and `grad` to create highly efficient, expressive code.
#
# ## Learning Objectives
#
# By the end of this notebook, you will:
# - Understand why manual batching is tedious and error-prone
# - Use `jax.vmap` to automatically vectorize functions
# - Control vectorization with `in_axes` and `out_axes`
# - Compose `vmap`, `jit`, and `grad` for maximum performance
# - Apply vectorization to compute Keplerian orbits

# %%
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import matplotlib.pyplot as plt
from time import time

# Check available devices
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# %% [markdown]
# ---
#
# ## 3.1 Automatic Vectorization with `jax.vmap`
#
# **Problem**: Manual batching is tedious and error-prone.
#
# Imagine you have a function that generates a single image — say, a 2D Gaussian. Now you want to generate 1,000 such images with different parameters.
#
# You have three options:
# 1. **Naive Python loop** — slow, especially on GPU
# 2. **Manual broadcasting** — fast but error-prone and requires rewriting your function
# 3. **`jax.vmap`** — automatic, fast, and composable
#
# We'll compare the loop to `vmap` directly; manual broadcasting is essentially the same math as `vmap`, but it usually forces you to restructure code as soon as the function gets more complicated.

# %% [markdown]
# ### Example: Generating 2D Gaussian Images
#
# Let's say we want to generate a batch of 2D Gaussian images, each with a different center and width.
# This is a common task in image processing and generative modeling.
#
# First, let's define the coordinate grid and a function to generate a *single* image.

# %%
# Define a grid of (x, y) coordinates
x = jnp.linspace(-3, 3, 100)
y = jnp.linspace(-3, 3, 100)
X, Y = jnp.meshgrid(x, y)

def gaussian_image_single(mu, sigma):
    """
    Generate a single 2D Gaussian image.
    
    Args:
        mu: center of the Gaussian (scalar) - for simplicity we shift only in x
        sigma: width of the Gaussian (scalar)
    
    Returns:
        image: 2D array of shape (100, 100)
    """
    return jnp.exp(-((X - mu)**2 + Y**2) / (2 * sigma**2))

# Test: Generate one image
img = gaussian_image_single(0.0, 1.0)
print(f"Single image shape: {img.shape}")

# %% [markdown]
# ### Approach 1: Naive Python Loop
#
# Now let's generate 1,000 images with different centers and widths using a Python loop:

# %%
# Generate 1,000 random parameters
key = jax.random.key(0)
n_images = 1000
mus = jax.random.normal(key, (n_images,))
sigmas = jnp.exp(jax.random.normal(key, (n_images,)))  # Log-normal distribution for sigma

# Approach 1: Python loop
def generate_images_loop(mus, sigmas):
    images = []
    for mu, sigma in zip(mus, sigmas):
        images.append(gaussian_image_single(mu, sigma))
    return jnp.stack(images)

# Time it
start = time()
images_loop = generate_images_loop(mus, sigmas)
time_loop = time() - start

print(f"Loop approach: {time_loop:.4f} seconds")
print(f"Output shape: {images_loop.shape}")


# %% [markdown]
# ### Approach 2: Automatic Vectorization with `jax.vmap`
#
# Manually rewriting functions to broadcast over batches can be fast, but it scales poorly: you end up reshaping/broadcasting arguments by hand, and the code stops looking like the single-system physics you started with.
#
# **`jax.vmap` solves this**: it automatically transforms a function that works on one example into one that works on a batch, without changing the original logic.

# %%
# Approach 2: vmap - automatic vectorization
# We use the ORIGINAL single-example function!
gaussian_images_batch = vmap(gaussian_image_single, in_axes=(0, 0), out_axes=0)

# Warm up (trigger compilation/caching)
_ = gaussian_images_batch(mus, sigmas).block_until_ready()

# Time it
start = time()
images_vmap = gaussian_images_batch(mus, sigmas).block_until_ready()
time_vmap = time() - start

print("image vmap shape:", images_vmap.shape)
print(f"vmap approach: {time_vmap:.4f} seconds")
print(f"Speedup vs loop: {time_loop / time_vmap:.1f}x")
print(f"Results match: {jnp.allclose(images_loop, images_vmap)}")


# %% [markdown]
# ### Controlling Vectorization with `in_axes` and `out_axes`
#
# Often you want fine-grained control over which axes to vectorize:
#
# - `in_axes`: specifies which axis to map over for each input
#   - `0` means map over axis 0
#   - `None` means don't map (broadcast this argument)
#   - Can be a tuple for multiple arguments: `in_axes=(0, 1, None)`
#
# - `out_axes`: specifies which axis the output should be mapped over (default is 0)
#
# In the Gaussian image example above, we used:
#
# ```python
# gaussian_images_batch = vmap(gaussian_image_single, in_axes=(0, 0), out_axes=0)
# ```
#
# That tuple says: map over the first axis of `mu` and `sigma` (batches of parameters). For the output, we want a new axis at the front (axis 0) to represent the batch dimension.


# %% [markdown]
# ---
#
# ## 3.2 Composing Transformations: The Power of JAX
#
# The real magic of JAX is that transformations **compose freely**. You can combine `jit`, `grad`, and `vmap` in any order, and they just work.
#
# %% [markdown]
# ### `vmap(grad(fn))` — Per-Example Gradients
#
# This is incredibly useful for machine learning and optimization: compute gradients for each example in a batch independently.

# %%
# Example: fitting a power law to multiple datasets
def power_law_loss(params, x_data, y_data):
    """
    Loss function for fitting y = A * x^alpha.
    
    Args:
        params: [A, alpha]
        x_data, y_data: observed data points
    
    Returns:
        loss: mean squared error
    """
    A, alpha = params
    y_pred = A * x_data ** alpha
    return jnp.mean((y_pred - y_data)**2)

# Generate synthetic datasets with different parameters
n_datasets = 5
n_points = 20

key = jax.random.key(42)
x_data = jnp.linspace(1, 10, n_points)

# True parameters for each dataset
true_params = jnp.array([
    [1.0, 2.0],
    [2.0, 1.5],
    [0.5, 2.5],
    [1.5, 1.8],
    [0.8, 2.2]
])

# Generate noisy data for each dataset
y_data_list = []
for i in range(n_datasets):
    A, alpha = true_params[i]
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (n_points,)) * 0.5
    y_data_list.append(A * x_data ** alpha + noise)

y_data_batch = jnp.array(y_data_list)  # shape: (5, 20)

# Initial guess for all datasets
params_init = jnp.array([1.0, 2.0])

# Compute gradient for a single dataset
grad_fn = grad(power_law_loss)
grad_single = grad_fn(params_init, x_data, y_data_batch[0])
print(f"Gradient for first dataset: {grad_single}")

# Compute gradients for ALL datasets at once using vmap
grad_batch_fn = vmap(grad_fn, in_axes=(None, None, 0))
grads_batch = grad_batch_fn(params_init, x_data, y_data_batch)

print(f"Gradients for all datasets:")
print(f"Shape: {grads_batch.shape}")  # (5, 2)
print(f"Values:")
print(f"{grads_batch}")

# %% [markdown]
# ### `jit(vmap(grad(fn)))` — Compile The Whole Batch

# %%
grad_batch_jit = jit(grad_batch_fn)
grads_batch_jit = grad_batch_jit(params_init, x_data, y_data_batch)
print(f"jit(vmap(grad)) matches: {jnp.allclose(grads_batch, grads_batch_jit)}")

# %% [markdown]
# ### Why Composability Matters
#
# The key insight is this: **you write single-example logic, and JAX handles the rest**.
#
# - Write a function for one planet → `vmap` handles 10,000 planets
# - Write a loss for one dataset → `vmap(grad)` computes gradients for all datasets
# - Add `jit` anywhere in the chain → instant compilation
#
# This is fundamentally different from libraries like NumPy where you must manually vectorize everything, or PyTorch where batching is often baked into the API.

# %% [markdown]
# ---
#
# ## 3.3 Hands-On Exercise: Keplerian Orbits
#
# Now let's put everything together in a realistic astrophysics application: computing Keplerian orbits for a large batch of exoplanet systems.
#
# ### The Physics
#
# For an elliptical orbit, the position $(x, y)$ at time $t$ is given by:
#
# 1. Solve Kepler's equation for the eccentric anomaly $E$:
#    $$M = E - e \sin E$$
#    where $M = n(t - t_0)$ is the mean anomaly and $n = 2\pi/T$ is the mean motion.
#
# 2. Compute the true anomaly $\nu$:
#    $$\nu = 2 \arctan\left(\sqrt{\frac{1+e}{1-e}} \tan\frac{E}{2}\right)$$
#
# 3. Compute the orbital radius:
#    $$r = a(1 - e \cos E)$$
#
# 4. Convert to Cartesian coordinates:
#    $$x = r \cos \nu, \quad y = r \sin \nu$$
#
# Let's implement this for a single orbit first, then vectorize it.

# %%
# Physical constants (in SI units)
G = 6.67430e-11  # m^3 kg^-1 s^-2
M_sun = 1.989e30  # kg
AU = 1.496e11     # meters

def solve_kepler(M, e, max_iter=50):
    """
    Solve Kepler's equation M = E - e*sin(E) for E using Newton-Raphson.
    
    Args:
        M: mean anomaly (radians)
        e: eccentricity
        max_iter: maximum iterations
    
    Returns:
        E: eccentric anomaly (radians)
    """
    # Initial guess: use jnp.where for vectorization
    E = jnp.where(e < 0.8, M, jnp.pi)
    
    # Newton-Raphson iteration
    # We use a fixed number of iterations for JIT simplicity
    # To improve: use jax.lax.while_loop to stop iteration automatically when convergence is reached
    # Note: Since max_iter is small (50), JAX will unroll this loop during compilation, which is fine.
    for _ in range(max_iter):
        f = E - e * jnp.sin(E) - M
        f_prime = 1 - e * jnp.cos(E)
        E = E - f / f_prime
    
    return E

def keplerian_orbit_single(t, a, e, T, t0=0.0):
    """
    Compute position for a single Keplerian orbit at time t.
    
    Args:
        t: time (seconds)
        a: semi-major axis (meters)
        e: eccentricity (0 <= e < 1)
        T: orbital period (seconds)
        t0: time of periapsis passage (seconds)
    
    Returns:
        x, y: position in orbital plane (meters)
    """
    # Mean motion
    n = 2 * jnp.pi / T
    
    # Mean anomaly
    M = n * (t - t0)
    M = M % (2 * jnp.pi)  # Wrap to [0, 2π]
    
    # Solve for eccentric anomaly (Newton-Raphson)
    E = solve_kepler(M, e)
    
    # True anomaly
    nu = 2 * jnp.arctan2(
        jnp.sqrt(1 + e) * jnp.sin(E / 2),
        jnp.sqrt(1 - e) * jnp.cos(E / 2)
    )
    
    # Orbital radius
    r = a * (1 - e * jnp.cos(E))
    
    # Cartesian coordinates
    x = r * jnp.cos(nu)
    y = r * jnp.sin(nu)
    
    return x, y

# Test: Earth's orbit at t=0 (perihelion)
t_test = 0.0
a_earth = 1.0 * AU
e_earth = 0.0167  # Earth's eccentricity
T_earth = 365.25 * 24 * 3600  # seconds

x, y = keplerian_orbit_single(t_test, a_earth, e_earth, T_earth)
print(f"Earth's position at perihelion:")
print(f"x = {x/AU:.4f} AU, y = {y/AU:.4f} AU")
print(f"Distance from Sun: {jnp.sqrt(x**2 + y**2)/AU:.4f} AU")

# %% [markdown]
# ### Vectorizing Over Time for a Single System
#
# Let's use `vmap` to compute the full orbit of a single planet by vectorizing over time `t`.
#
# We have a function `keplerian_orbit_single(t, a, e, T, t0)` that computes the position at a single time `t`.
# To get the full orbit, we want to evaluate this function for many values of `t`.
#
# Instead of writing a loop, we can use `vmap`!

# %%
# Define parameters for a single planet (e.g., highly elliptical orbit)
a = 1.0 * AU        # Semi-major axis
e = 0.6             # High eccentricity to make the orbit interesting
T = 365.25 * 24 * 3600  # Period (1 year in seconds)
t0 = 0.0            # Time of periapsis passage

# Generate time points for one full orbit
n_points = 100
t_values = jnp.linspace(0, T, n_points)

# Use vmap to vectorize over time (the first argument, axis 0)
# in_axes=(0, None, None, None, None) means:
# - map over the 0-th axis of the first argument (t)
# - broadcast the other arguments (a, e, T, t0)
keplerian_orbit_over_time = vmap(keplerian_orbit_single, in_axes=(0, None, None, None, None))

# Compute the orbit
start = time()
x_orbit, y_orbit = keplerian_orbit_over_time(t_values, a, e, T, t0)
print(f"Computed {n_points} points in {time() - start:.4f} seconds")

# Verify shapes
print(f"x_orbit shape: {x_orbit.shape}")
print(f"y_orbit shape: {y_orbit.shape}")

# %%
# Plot the orbit
plt.figure(figsize=(8, 8))

# Plot the path
plt.plot(x_orbit/AU, y_orbit/AU, label=f'Orbit (e={e})', color='blue')

# Mark the star (focus)
plt.plot(0, 0, 'yo', markersize=15, markeredgecolor='orange', markeredgewidth=2, label='Star')

# Formatting
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('Keplerian Orbit Vectorized over Time with vmap')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Why this is powerful
#
# By using `vmap(keplerian_orbit_single, in_axes=(0, None, None, None, None))`, we transformed a function that operates on scalar time `t` into one that operates on a vector of times `t_values`.
#
# We didn't have to change `keplerian_orbit_single` at all! This separation of concerns—writing the physics for a single point, then vectorizing it—is a core philosophy of JAX.

# %% [markdown]
# ---
#
# ## Summary
#
# In this notebook, we explored `jax.vmap` and the power of composable transformations:
#
# ### Key Takeaways
#
# 1. **Manual batching is tedious**: Python loops are slow, and manual broadcasting requires rewriting functions.
#
# 2. **`vmap` automates vectorization**: Transform any single-example function into a batched version automatically.
#
# 3. **Control with `in_axes` and `out_axes`**: Fine-grained control over which dimensions to vectorize. In our orbit example, we vectorized over time (axis 0) while keeping other parameters fixed (axis `None`).
#
# 4. **Transformations compose freely**: 
#    - `jit(vmap(fn))` — vectorize and compile
#    - `vmap(grad(fn))` — per-example gradients
#
# 5. **Write single-example logic**: Let JAX handle batching, differentiation, and compilation.
#
# ### Next Steps
#
# In the next notebook, we'll explore JAX's functional programming patterns:
# - Pure functions and why they matter
# - PRNG key management
# - PyTrees for structured data
#
# These patterns will complete your JAX toolkit and prepare you for real-world astrophysics applications.

# %%
