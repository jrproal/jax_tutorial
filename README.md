# JAX for Astrophysics: A Hands-On Tutorial

**Authors:** Xiaoyue Cao [曹潇月] (Institute for Astrophysics, School of Physics, Zhengzhou University) & Gemini AI

A comprehensive tutorial introducing JAX to astrophysics PhD students, covering core transformations, functional programming patterns, and practical applications in scientific computing.

## Overview

This tutorial teaches you how to leverage JAX's powerful features—JIT compilation, automatic differentiation, and automatic vectorization—for astrophysics research.

**Target Audience:** Astrophysics PhD students with working knowledge of Python and NumPy.

**Estimated Duration:** 1.5–2 hours (reading + running code examples)

## Learning Objectives

By the end of this tutorial, you will be able to:

- Understand JAX's design philosophy and when to use it over NumPy/PyTorch/TensorFlow
- Use `jax.jit` to compile and accelerate your scientific computing code
- Compute exact gradients with `jax.grad` for optimization and inference
- Vectorize computations across parameter batches with `jax.vmap`
- Compose JAX transformations (`jit`, `grad`, `vmap`) for powerful workflows
- Write JAX-compatible code following functional programming principles
- Apply JAX to real astrophysics problems (model fitting, Bayesian inference)

## Prerequisites

- Working knowledge of Python
- Familiarity with NumPy for array operations
- Basic understanding of scientific computing concepts
- (Optional) Basic calculus for understanding automatic differentiation

## Setup Instructions

### 1. Create and activate the conda environment

```bash
# Create the environment
conda create -n jax_tutor python=3.11

# Activate the environment
conda activate jax_tutor
```

### 2. Install additional dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Verify JAX installation

Open any notebook and run:

```python
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Test basic computation
x = jnp.array([1.0, 2.0, 3.0])
print(f"Test array: {x}")
```

## Tutorial Structure

The tutorial consists of 4 Jupyter notebooks, each covering a self-contained topic:

### [01_introduction.ipynb](01_introduction.ipynb) (~15 min)
- Why JAX for astrophysics?
- JAX vs NumPy: Key differences (Immutability, float32)
- Structured Control Flow (`cond`, `loop`, `scan`)
- **Exercise:** Port a blackbody spectrum calculation to JAX

### [02_jit_and_autodiff.ipynb](02_jit_and_autodiff.ipynb) (~25 min)
- Just-In-Time compilation with `jax.jit`
- Automatic differentiation with `jax.grad`
- Higher-order derivatives (Jacobian, Hessian)
- **Exercise:** Gradient-based power-law model fitting

### [03_vmap_and_composability.ipynb](03_vmap_and_composability.ipynb) (~20 min)
- Automatic vectorization with `jax.vmap`
- Composing transformations: `jit(vmap(grad(fn)))`
- **Exercise:** Batch Keplerian orbit computation

### [04_functional_patterns.ipynb](04_functional_patterns.ipynb) (~20 min)
- Pure functions and JAX requirements
- PRNG key management with `jax.random`
- PyTrees for structured data
- Debugging tips and common pitfalls
- **Exercise:** Simple Metropolis-Hastings MCMC sampler

## Key Concepts

### JAX's Four Core Transformations

| Transformation | What It Does | Typical Use Case |
|---|---|---|
| `jax.jit` | Compiles function via XLA | Speed up repeated computations |
| `jax.grad` | Reverse-mode autodiff | Gradients for optimization/inference |
| `jax.vmap` | Auto-vectorization | Batched evaluation across parameters |
| `jax.lax.scan` | Sequential loop | Time integration, cumulative operations |

### JAX's Design Philosophy

JAX is built around **composable function transformations** over **pure functions**:

- **Functional purity:** No side effects, no hidden state
- **Composability:** Transformations can be freely nested (e.g., `jit(vmap(grad(fn)))`)
- **NumPy compatibility:** Familiar API for easy adoption
- **Hardware acceleration:** Seamless CPU/GPU/TPU execution

## Additional Resources

### Official Documentation
- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub Repository](https://github.com/jax-ml/jax)
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)

### JAX Ecosystem
- [Optax](https://optax.readthedocs.io/) - Gradient-based optimization
- [NumPyro](https://num.pyro.ai/) - Probabilistic programming and MCMC
- [Diffrax](https://docs.kidger.site/diffrax/) - Differential equation solvers
- [JAXopt](https://jaxopt.github.io/) - Hardware-accelerated optimization

### Astrophysics-Specific Libraries
- [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) - Differentiable cosmology
- [jaxoplanet](https://jax.exoplanet.codes/) - Exoplanet modeling
- [ripple](https://github.com/tedwards2412/ripple) - Gravitational wave analysis
- [astronomix](https://github.com/hsinfan1996/astronomix) - MHD simulations
- [jaxspec](https://jaxspec.readthedocs.io/) - X-ray spectroscopy
- [awesome-JAXtronomy](https://github.com/JAXtronomy/awesome-JAXtronomy) - Curated list of JAX in astronomy

### Tutorials and Guides
- [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Patrick Kidger's JAX Tutorials](https://docs.kidger.site/)
- [A brief introduction to JAX (jaxoplanet)](https://jax.exoplanet.codes/en/latest/tutorials/introduction-to-jax/)

## Getting Help

If you encounter issues or have questions:

1. Check the [JAX FAQ](https://jax.readthedocs.io/en/latest/faq.html)
2. Search [JAX GitHub Issues](https://github.com/jax-ml/jax/issues)
3. Ask on [JAX Discussions](https://github.com/jax-ml/jax/discussions)
4. Consult the [JAX documentation](https://jax.readthedocs.io/)

## License

This tutorial is provided for educational purposes. Feel free to use and adapt it for your research and teaching.

## Acknowledgments

This tutorial draws inspiration from:
- The official JAX documentation and tutorials
- The JAXtronomy community
- Patrick Kidger's excellent JAX educational materials
- The broader scientific Python community
