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
# # Notebook 4 — JAX Programming Patterns & Gotchas
#
# **Estimated time:** ~20 minutes
#
# In this notebook, we'll explore the essential programming patterns that make JAX work. Understanding these patterns is crucial for writing correct, efficient, and maintainable JAX code.
#
# ## What You'll Learn
#
# 1. **Pure Functions** — Why JAX requires functional programming and common pitfalls
# 2. **Random Number Generation** — JAX's explicit PRNG key system
# 3. **PyTrees** — JAX's universal data structure for organizing complex data
# 4. **Debugging Tips** — Tools and techniques for debugging JAX code
#
# Let's start by importing the necessary libraries.

# %%
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import matplotlib.pyplot as plt

# Check JAX version and available devices
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# %% [markdown]
# ---
#
# ## 4.1 Pure Functions: The Golden Rule
#
# ### What is a Pure Function?
#
# A **pure function** is a function that:
# 1. **Has no side effects** — doesn't modify external state, doesn't perform I/O
# 2. **Is deterministic** — same inputs always produce same outputs
# 3. **Doesn't depend on hidden state** — no global variables, no mutable state
#
# ### Why JAX Requires Purity
#
# JAX's transformations (`jit`, `grad`, `vmap`) work by **tracing** your function with abstract values. During tracing:
# - JAX builds a computation graph
# - Side effects may not execute as expected
# - Mutable state can lead to incorrect results
#
# Let's see some common mistakes and how to fix them.

# %% [markdown]
# ### Common Mistake #1: Global Mutable State

# %%
# ❌ BAD: Using global mutable state
counter = 0

def bad_compute_energy(mass, velocity):
    global counter
    counter += 1  # Side effect!
    return 0.5 * mass * velocity**2

# This will not work as expected with JIT
jit_bad_energy = jit(bad_compute_energy)

print("Before JIT calls, counter:", counter)
result1 = jit_bad_energy(1.0, 2.0)
result2 = jit_bad_energy(1.0, 3.0)
print("After 2 JIT calls, counter:", counter)  # Counter only incremented once during tracing!


# %%
# ✅ GOOD: Pure function without side effects
def good_compute_energy(mass, velocity):
    return 0.5 * mass * velocity**2

jit_good_energy = jit(good_compute_energy)

result1 = jit_good_energy(1.0, 2.0)
result2 = jit_good_energy(1.0, 3.0)
print(f"Energy 1: {result1:.2f}, Energy 2: {result2:.2f}")


# %% [markdown]
# ### Common Mistake #2: Print Statements Inside JIT

# %%
# ❌ BAD: Regular print inside JIT
@jit
def bad_gravitational_force(m1, m2, r):
    G = 6.674e-11  # Gravitational constant
    print(f"Computing force for r={r}")  # Only prints during tracing!
    return G * m1 * m2 / r**2

print("First call:")
f1 = bad_gravitational_force(1e30, 1e30, 1e11)
print("\nSecond call:")
f2 = bad_gravitational_force(1e30, 1e30, 2e11)  # No print output!

# %% [markdown]
# ✅ GOOD: Use `jax.debug.print` for printing inside JIT:
#
# ```python
# @jit
# def good_gravitational_force(m1, m2, r):
#     G = 6.674e-11
#     jax.debug.print("Computing force for r={}", r)
#     return G * m1 * m2 / r**2
# ```

# %% [markdown]
# ### Common Mistake #3: In-Place Mutation

# %%
# ❌ BAD: Trying to mutate arrays in-place
try:
    positions = jnp.array([1.0, 2.0, 3.0])
    positions[0] = 10.0  # This will raise an error!
except TypeError as e:
    print(f"Error: {e}")

# %%
# ✅ GOOD: Use .at[].set() for functional updates
positions = jnp.array([1.0, 2.0, 3.0])
positions_updated = positions.at[0].set(10.0)  # Returns a new array

print("Original:", positions)
print("Updated:", positions_updated)

# %% [markdown]
# ---
#
# ## 4.2 Random Number Generation: Explicit PRNG Keys
#
# ### Why JAX Doesn't Use Global State
#
# NumPy's random number generation uses a global state:
#
# ```python
# import numpy as np
# np.random.seed(42)
# x = np.random.normal()  # Uses global state
# ```
#
# This approach has problems:
# - **Not thread-safe** — parallel execution can give non-deterministic results
# - **Hard to reproduce** — hidden state makes debugging difficult
# - **Breaks functional purity** — violates JAX's design principles
#
# ### JAX's Solution: Explicit PRNG Keys
#
# JAX uses **explicit PRNG keys** that you pass around explicitly. This makes randomness:
# - **Deterministic** — same key always produces same random numbers
# - **Reproducible** — easy to debug and test
# - **Parallelizable** — different keys can be used independently

# %%
# Creating a PRNG key
key = jax.random.key(42)  # Modern API (JAX >= 0.4.16)
# or: key = jax.random.PRNGKey(42)  # Older API, still works

print("PRNG key:", key)
print("Key shape:", key.shape)
print("Key dtype:", key.dtype)

# %% [markdown]
# ### Splitting Keys: The Core Pattern
#
# The key operation in JAX random number generation is **splitting** keys. This creates new independent keys from an existing one:

# %%
# Split a key into two new keys
key = jax.random.key(0)
key1, key2 = jax.random.split(key)

print("Original key:", key)
print("New key 1:", key1)
print("New key 2:", key2)

# Generate random numbers with each key
x1 = jax.random.normal(key1)
x2 = jax.random.normal(key2)

print(f"\nRandom value from key1: {x1:.4f}")
print(f"Random value from key2: {x2:.4f}")

# Using the same key again gives the same result (deterministic!)
x1_again = jax.random.normal(key1)
print(f"Using key1 again: {x1_again:.4f} (same as before!)")

# %% [markdown]
# ### Splitting Multiple Keys at Once

# %%
# Split into multiple keys at once
key = jax.random.key(42)
keys = jax.random.split(key, num=5)  # Create 5 new keys

print("Generated 5 keys:")
for i, k in enumerate(keys):
    print(f"  Key {i}: {k}")


# %% [markdown]
# ### Pattern: Threading Keys Through Computation
#
# The standard pattern is to split your key before each random operation:

# %%
def monte_carlo_pi_estimate(key, n_samples):
    """
    Estimate π using Monte Carlo method.
    
    We generate random points in a unit square and count how many
    fall inside a quarter circle.
    """
    # Split key for x and y coordinates
    key_x, key_y = jax.random.split(key)
    
    # Generate random points
    x = jax.random.uniform(key_x, shape=(n_samples,))
    y = jax.random.uniform(key_y, shape=(n_samples,))
    
    # Check if points are inside quarter circle
    inside_circle = (x**2 + y**2) <= 1.0
    
    # Estimate π
    pi_estimate = 4.0 * jnp.mean(inside_circle)
    return pi_estimate

# Run the estimation
key = jax.random.key(123)
pi_est = monte_carlo_pi_estimate(key, 1_000_000)
print(f"π estimate: {pi_est:.6f}")
print(f"True π: {jnp.pi:.6f}")
print(f"Error: {abs(pi_est - jnp.pi):.6f}")

# %% [markdown]
# ### Common Random Functions

# %%
key = jax.random.key(0)
key, key_uniform = jax.random.split(key)
uniform_samples = jax.random.uniform(key_uniform, shape=(5,))
print("Uniform [0,1):", uniform_samples)

key, key_normal = jax.random.split(key)
normal_samples = jax.random.normal(key_normal, shape=(5,))
print("Normal(0,1):", normal_samples)

key, key_int = jax.random.split(key)
int_samples = jax.random.randint(key_int, shape=(5,), minval=0, maxval=10)
print("Random ints [0,10):", int_samples)

key, key_choice = jax.random.split(key)
array = jnp.array([10, 20, 30, 40, 50])
choices = jax.random.choice(key_choice, array, shape=(3,))
print("Random choices:", choices)


# %% [markdown]
# ---
#
# ## 4.3 PyTrees: JAX's Universal Data Structure
#
# ### What is a PyTree?
#
# A **PyTree** (Python tree) is JAX's term for a nested structure of containers (lists, tuples, dicts) with arrays at the leaves.
#
# Examples of PyTrees:
# - A single array: `jnp.array([1, 2, 3])`
# - A list of arrays: `[array1, array2, array3]`
# - A dict of arrays: `{"mass": array1, "velocity": array2}`
# - Nested structures: `{"params": {"w": array1, "b": array2}, "state": array3}`
# - Named tuples with arrays
#
# ### Why PyTrees Matter
#
# All JAX transformations (`jit`, `grad`, `vmap`, etc.) work seamlessly with PyTrees:
# - You can pass complex nested structures to JIT-compiled functions
# - `grad` can differentiate with respect to PyTree parameters
# - `vmap` can vectorize over PyTree inputs
#
# This makes organizing complex data (like model parameters) much easier!

# %% [markdown]
# ### Basic PyTree Operations

# %%
# Example: organizing model parameters as a PyTree
params = {
    "amplitude": jnp.array(1.5),
    "frequency": jnp.array(2.0),
    "phase": jnp.array(0.5),
}

print("Parameters (PyTree):")
for key, value in params.items():
    print(f"  {key}: {value}")


# %% [markdown]
# ### Using PyTrees with JAX Transformations

# %%
# Define a function that takes a PyTree as input
def sinusoidal_model(params, t):
    """A simple sinusoidal model: A * sin(ω*t + φ)"""
    return params["amplitude"] * jnp.sin(params["frequency"] * t + params["phase"])

# This works with JIT!
@jit
def jit_model(params, t):
    return sinusoidal_model(params, t)

t = jnp.linspace(0, 2*jnp.pi, 100)
y = jit_model(params, t)

print(f"Model output shape: {y.shape}")

plt.figure(figsize=(8, 4))
plt.plot(t, y)
plt.title("Sinusoidal Model Output")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# %%
# Gradients work with PyTrees too!
def loss_function(params, t, y_true):
    """Mean squared error loss"""
    y_pred = sinusoidal_model(params, t)
    return jnp.mean((y_pred - y_true)**2)

# Compute gradient with respect to all parameters
grad_fn = jit(grad(loss_function, argnums=0)) #grad_fn = jit(grad(loss_function, argnums=0))

# Generate some synthetic data
t_data = jnp.linspace(0, 2*jnp.pi, 50)
y_data = 2.0 * jnp.sin(3.0 * t_data + 0.1)  # True params: A=2, ω=3, φ=0.1

# Compute gradients
grads = grad_fn(params, t_data, y_data)

print("Gradients (also a PyTree):")
for key, value in grads.items():
    print(f"  ∂loss/∂{key}: {value:.6f}")

# %% [markdown]
# ### jax.tree.map: Apply Functions to Every Leaf

# %%
# tree.map applies a function to every leaf in a PyTree

# Example 1: Scale all parameters by 2
scaled_params = jax.tree.map(lambda x: 2 * x, params)
print("Original parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")
print("\nScaled parameters:")
for key, value in scaled_params.items():
    print(f"  {key}: {value}")

# Example 2: Gradient descent update
learning_rate = 0.1
updated_params = jax.tree.map(
    lambda p, g: p - learning_rate * g,
    params, grads
)
print("\nUpdated parameters (after one gradient step):")
for key, value in updated_params.items():
    print(f"  {key}: {value:.6f}")

# %% [markdown]
# ### Registering Custom Classes as PyTrees
#
# You can register your own classes to work as PyTrees:

# %%
@jax.tree_util.register_pytree_node_class
class SimpleModel:
    """A basic linear model registered as a JAX PyTree.
    
    Registering as a PyTree allows JAX transformations (jit, grad, vmap) to
    handle this object and distinguish between parameters and metadata.
    """
    def __init__(self, w: jax.Array, b: jax.Array, model_id: str):
        self.w = w
        self.b = b
        self.model_id = model_id

    def tree_flatten(self):
        """Flattens the object into children and auxiliary data.
        
        Returns:
            tuple: (children, aux_data)
            children: JAX arrays or other PyTrees to be traced/differentiated.
            aux_data: Static metadata (strings, bools, etc.) that stays constant.
        """
        children = (self.w, self.b)  # Parameters to be optimized
        aux_data = self.model_id      # Metadata that doesn't change
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstructs the object from auxiliary data and children.
        
        Args:
            aux_data: Static metadata from tree_flatten.
            children: Transformed children from tree_flatten.
        """
        w, b = children
        return cls(w, b, aux_data)

# Initialize a simple model
model = SimpleModel(
    w=jnp.array([1.0, 2.0, 3.0]),
    b=jnp.array(0.5),
    model_id="v1",
)

print(f"Model ID: {model.model_id}")
print(f"Weights: {model.w}")
print(f"Bias: {model.b}")

@jit
def forward(model: SimpleModel, x: jax.Array) -> jax.Array:
    """Computes the forward pass of the model.
    
    JAX's @jit handles the model instance because it's a registered PyTree.
    """
    return x @ model.w + model.b

# Test the model
x = jnp.array([1.0, 1.0, 1.0])
prediction = forward(model, x)
print(f"\nPrediction: {prediction:.4f}")

# %% [markdown]
# ### JIT Compiling Class Methods
#
# There are two main ways to use `@jit` on methods within a class:
# 1. **`static_argnums=(0,)`**: Treat `self` as a static argument (constant during tracing). This causes a re-compile every time a different instance is used.
# 2. **PyTree Registration**: Register the class as a PyTree. This allows `self` to be treated as data, enabling JIT to work across different instances without unnecessary re-compilation.

# %%
# Method 1: Using static_argnums=(0,)
class Scaler:
    def __init__(self, factor):
        self.factor = factor

    @jax.jit(static_argnums=(0,))
    def multiply(self, x):
        # self is treated as a static constant during JIT compilation
        return x * self.factor

scaler = Scaler(2.0)
print(f"Static JIT result: {scaler.multiply(10.0)}")

# %%
# Method 2: Using PyTree Registration (Recommended)
@jax.tree_util.register_pytree_node_class
class PyTreeScaler:
    def __init__(self, factor):
        self.factor = factor

    def tree_flatten(self):
        return (self.factor,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jax.jit
    def multiply(self, x):
        # self is now a valid PyTree node, so JIT can trace its attributes
        return x * self.factor

pt_scaler = PyTreeScaler(3.0)
print(f"PyTree JIT result: {pt_scaler.multiply(10.0)}")


# %% [markdown]
# ---
#
# ## 4.4 Debugging Tips
#
# Debugging JAX code can be tricky because of JIT compilation and tracing. Here are essential tools and techniques.

# %% [markdown]
# ### Tool 1: jax.debug.print for Printing Inside JIT
#
# Regular `print()` only executes during tracing. Use `jax.debug.print()` to print actual values:

# %%
@jit
def compute_with_debug(x, y):
    # Regular print - only during tracing
    print(f"[TRACE] Function called with shapes: x={x.shape}, y={y.shape}")
    
    # Debug print - prints actual values at runtime
    jax.debug.print("[RUNTIME] x = {}, y = {}", x, y)
    
    result = x ** 2 + y ** 2
    jax.debug.print("[RUNTIME] result = {}", result)
    
    return result

print("First call:")
out1 = compute_with_debug(jnp.array(2.0), jnp.array(3.0))

print("\nSecond call (different values):")
out2 = compute_with_debug(jnp.array(5.0), jnp.array(7.0))


# %% [markdown]
# ### Tool 2: jax.disable_jit() for Debugging
#
# Temporarily disable JIT compilation to run in eager mode (like regular NumPy):

# %%
@jit
def buggy_function(x):
    # ❌ This will fail under @jit because x is a Tracer during tracing
    if x > 0:
        return x
    return -x

# With JIT, error messages can be long and cryptic
print("1. Running with JIT (will fail during tracing):")
try:
    result = buggy_function(jnp.array(1.0))
except Exception as e:
    print(f"Caught JIT error: {type(e).__name__}")
    # print(f"Error message: {str(e)[:100]}...")

# Disable JIT for clearer debugging
print("\n2. Running with jax.disable_jit() (runs in eager mode):")
with jax.disable_jit():
    try:
        result = buggy_function(jnp.array(1.0))
        print(f"Eager mode result: {result}")
    except Exception as e:
        print(f"Eager mode error: {type(e).__name__}")
        print(f"Message: {str(e)[:100]}")


# %% [markdown]
# ### Tool 3: Check Shapes and Dtypes
#
# Many JAX errors come from shape or dtype mismatches. Always check:

# %%
def debug_array_info(name, arr):
    """Print detailed info about an array"""
    print(f"{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  devices: {arr.devices()}")
    print(f"  min/max: {arr.min():.4f} / {arr.max():.4f}")
    print(f"  mean/std: {arr.mean():.4f} / {arr.std():.4f}")
    print()

# Example
positions = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
velocities = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

debug_array_info("positions", positions)
debug_array_info("velocities", velocities)

# %% [markdown]
# ### Common Error 1: Shape Mismatch

# %%
# ❌ Shape mismatch
try:
    a = jnp.array([1, 2, 3])
    b = jnp.array([[1, 2], [3, 4]])
    result = a + b  # Shapes (3,) and (2, 2) don't broadcast!
except Exception as e:
    print(f"Error: {type(e).__name__}")
    print(f"Message: {e}")

# ✅ Check shapes first
print(f"\na.shape = {a.shape}")
print(f"b.shape = {b.shape}")
print("These shapes are incompatible for broadcasting!")

# %% [markdown]
# ### Common Error 2: Dtype Issues

# %%
# JAX defaults to float32, which can cause precision issues
x_float32 = jnp.array(1.0)  # Default: float32
x_float64 = jnp.array(1.0, dtype=jnp.float64)

print(f"float32: {x_float32.dtype}")
print(f"float64: {x_float64.dtype}")

# Enable float64 globally if needed
from jax import config
config.update("jax_enable_x64", True)

x_now = jnp.array(1.0)
print(f"After enabling x64: {x_now.dtype}")

# Disable it again for this tutorial
config.update("jax_enable_x64", False)


# %% [markdown]
# ### Common Error 3: Traced Value in Conditional

# %%
# ❌ This fails
@jit
def bad_branch(x):
    if x > 0:  # Can't branch on traced value!
        return x
    else:
        return -x

try:
    result = bad_branch(jnp.array(1.0))
except Exception as e:
    print(f"Error: {type(e).__name__}")
    print("Solution: Use jax.lax.cond instead!")


# %% [markdown]
# ### Debugging Strategy: Start Simple, Add Complexity
#
# 1. **Test without JIT first** — Make sure your function works in eager mode
# 2. **Add JIT gradually** — JIT individual functions, not everything at once
# 3. **Use small test cases** — Debug with tiny arrays before scaling up
# 4. **Check intermediate values** — Use `jax.debug.print` liberally
# 5. **Verify gradients numerically** — Compare `grad` to finite differences

# %%
# Example: Verify gradient numerically
def f(x):
    return jnp.sum(x ** 3)

x = jnp.array([1.0, 2.0, 3.0])

# Analytical gradient (via JAX)
grad_analytical = grad(f)(x)

# Numerical gradient (finite differences)
epsilon = 1e-3
grad_numerical = jnp.zeros_like(x)
for i in range(len(x)):
    x_plus = x.at[i].set(x[i] + epsilon)
    x_minus = x.at[i].set(x[i] - epsilon)
    grad_numerical = grad_numerical.at[i].set((f(x_plus) - f(x_minus)) / (2 * epsilon))

print("Analytical gradient:", grad_analytical)
print("Numerical gradient:", grad_numerical)
print("Difference:", jnp.abs(grad_analytical - grad_numerical))
print("\nGradients match! ✓" if jnp.allclose(grad_analytical, grad_numerical, atol=1e-2) else "Gradients don't match! ✗")


# %% [markdown]
# ### Reading JAX Error Messages
#
# JAX error messages can be verbose. Key things to look for:
#
# 1. **ConcretizationTypeError** — You're trying to use a traced value in a Python control flow
#    - Solution: Use `jax.lax.cond`, `jax.lax.scan`, etc.
#
# 2. **TypeError: 'ArrayImpl' object does not support item assignment** — You're trying to mutate an array
#    - Solution: Use `.at[].set()` instead
#
# 3. **Shape mismatch** — Arrays have incompatible shapes
#    - Solution: Check shapes with `.shape`, use `jnp.reshape` or broadcasting
#
# 4. **Dtype mismatch** — Arrays have incompatible dtypes
#    - Solution: Cast explicitly with `.astype()` or enable x64
#
# 5. **TracerBoolConversionError** — You're using a traced array in a boolean context
#    - Solution: Use `jax.lax.cond` or `jnp.where`


# %% [markdown]
# ## Call external functions (from scipy for example) within jax function
#
# JAX's transformations like `jax.jit` and `jax.grad` only work on pure JAX functions. If you need to call an external function (like a standard Python function or a Scipy function) within a JIT-compiled function, you can use `jax.pure_callback`.

# %%
import scipy.special

def scipy_gamma(x):
    """Call scipy.special.gamma using jax.pure_callback"""
    # jax.pure_callback allows us to run Python functions inside JIT-compiled code
    return jax.pure_callback(
        scipy.special.gamma,                   # The external function
        jax.ShapeDtypeStruct(x.shape, x.dtype), # Shape and dtype of the output
        x                                      # Arguments
    )

@jax.jit
def jitted_function(x):
    # This works because of pure_callback!
    return scipy_gamma(x) + 1.0

x_val = jnp.array([1.0, 2.0, 3.0])
result = jitted_function(x_val)
print(f"Input: {x_val}")
print(f"Result (gamma(x) + 1): {result}")
print(f"Result Scipy (gamma(x) + 1): {scipy_gamma(x_val) + 1.0}")


# %% [markdown]
# ---
#
# ## Summary
#
# In this notebook, we covered essential JAX programming patterns:
#
# 1. **Pure Functions** — JAX requires functional programming (no side effects, no mutable state)
# 2. **Random Number Generation** — Use explicit PRNG keys with `jax.random.key()` and `jax.random.split()`
# 3. **PyTrees** — Organize complex data as nested dicts/lists/tuples; all JAX transformations work seamlessly
# 4. **Debugging** — Use `jax.debug.print`, `jax.disable_jit()`, and verify shapes/dtypes
#
# These patterns are fundamental to writing correct, efficient JAX code. In the next notebook, we'll apply everything we've learned to real astrophysics applications!

# %% [markdown]
# ### Exercise: Implement a Simple MCMC Sampler
#
# As a final exercise, try implementing a simple Metropolis-Hastings MCMC sampler using the patterns we've learned (and `jax.lax.scan` from Notebook 1):
#
# - Use `jax.random.split` to manage PRNG keys
# - Use `jax.lax.scan` to iterate through MCMC steps
# - Use PyTrees to organize sampler state
# - Make it JIT-compilable for speed
#
# Here's a reference implementation:

# %%
@jit(static_argnums=(1, 3))
def metropolis_hastings(key, log_prob_fn, initial_params, n_steps, step_size=0.1):
    """
    Simple Metropolis-Hastings MCMC sampler.
    
    Args:
        key: PRNG key
        log_prob_fn: Function that computes log probability
        initial_params: Starting parameters (PyTree)
        n_steps: Number of MCMC steps
        step_size: Proposal step size
    
    Returns:
        samples: Array of samples (n_steps, ...)
        acceptance_rate: Fraction of accepted proposals
    """
    def step(state, _):
        params, log_prob, key, n_accepted = state
        
        # Split key for proposal and acceptance
        key, key_proposal, key_accept = jax.random.split(key, 3)
        
        # Split key_proposal for each leaf to avoid correlation
        leaves, treedef = jax.tree_util.tree_flatten(params)
        keys_leaves = jax.random.split(key_proposal, len(leaves))
        keys_tree = jax.tree_util.tree_unflatten(treedef, keys_leaves)
        
        # Propose new parameters (random walk)
        proposal = jax.tree.map(
            lambda p, k: p + step_size * jax.random.normal(k, p.shape),
            params, keys_tree
        )
        
        # Compute log probability of proposal
        log_prob_proposal = log_prob_fn(proposal)
        
        # Accept/reject
        log_accept_ratio = log_prob_proposal - log_prob
        accept = jnp.log(jax.random.uniform(key_accept)) < log_accept_ratio
        
        # Update state
        new_params = jax.lax.cond(
            accept,
            lambda _: proposal,
            lambda _: params,
            None
        )
        new_log_prob = jax.lax.cond(
            accept,
            lambda _: log_prob_proposal,
            lambda _: log_prob,
            None
        )
        new_n_accepted = n_accepted + accept.astype(jnp.int32)
        
        new_state = (new_params, new_log_prob, key, new_n_accepted)
        output = new_params
        
        return new_state, output
    
    # Initialize
    initial_log_prob = log_prob_fn(initial_params)
    initial_state = (initial_params, initial_log_prob, key, 0)
    
    # Run MCMC
    final_state, samples = jax.lax.scan(step, initial_state, jnp.arange(n_steps))
    
    _, _, _, n_accepted = final_state
    acceptance_rate = n_accepted / n_steps
    
    return samples, acceptance_rate

# Test with a simple 2D Gaussian
def log_prob_gaussian(params):
    """Log probability of 2D Gaussian centered at origin"""
    x, y = params["x"], params["y"]
    return -0.5 * (x**2 + y**2)

initial_params = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
key = jax.random.key(42)

samples, accept_rate = metropolis_hastings(
    key, log_prob_gaussian, initial_params, n_steps=20000, step_size=1.0
)

print(f"Acceptance rate: {accept_rate:.2%}")
print(f"Mean x: {jnp.mean(samples['x']):.3f} (expected: 0.0)")
print(f"Mean y: {jnp.mean(samples['y']):.3f} (expected: 0.0)")
print(f"Std x: {jnp.std(samples['x']):.3f} (expected: 1.0)")
print(f"Std y: {jnp.std(samples['y']):.3f} (expected: 1.0)")

# %%
# Visualize samples
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Trace plot
axes[0].plot(samples['x'], alpha=0.5, label='x')
axes[0].plot(samples['y'], alpha=0.5, label='y')
axes[0].set_xlabel('MCMC Step')
axes[0].set_ylabel('Parameter Value')
axes[0].set_title('MCMC Trace')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2D scatter
axes[1].scatter(samples['x'][::10], samples['y'][::10], alpha=0.3, s=1)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('2D Posterior Samples')
axes[1].axis('equal')
axes[1].grid(True, alpha=0.3)

# Add circle showing true 1-sigma contour
theta = jnp.linspace(0, 2*jnp.pi, 100)
axes[1].plot(jnp.cos(theta), jnp.sin(theta), 'r--', linewidth=2, label='True 1σ')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# Congratulations! You've now mastered the essential functional programming patterns in JAX. You're ready to tackle real astrophysics applications in the next notebook!
