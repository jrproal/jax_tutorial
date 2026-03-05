# Convert the example code in `exercise.py` to JAX and save the new version as `exercise_jax.py`. Please apply JAX’s `jit` compilation to maximize GPU performance.

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Finds a root of f(x) = 0 using the Newton-Raphson method:
    x_{n+1} = x_n - f(x_n) / f'(x_n)
    
    Args:
        f: The function to find the root of.
        df: The derivative of f.
        x0: Initial guess.
        tol: Tolerance for convergence (stop when |f(x)| < tol).
        max_iter: Maximum number of iterations.
        
    Returns:
        The approximate root, or None if failed.
    """
    x = x0
    print(f"Starting Newton's method with initial guess: {x0}")
    
    for i in range(max_iter):
        fx = f(x)
        
        # Check for convergence
        if abs(fx) < tol:
            print(f"Converged after {i} iterations.")
            return x
        
        dfx = df(x)
        
        # Avoid division by zero
        if dfx == 0:
            print("Error: Derivative is zero. Cannot continue.")
            return None
            
        # Update rule: x_new = x_old - f(x) / f'(x)
        x_new = x - fx / dfx
        
        print(f"Iteration {i+1}: x = {x_new:.8f}, f(x) = {fx:.8f}")
        x = x_new
        
    print("Max iterations reached without convergence.")
    return x

def main():
    print("=== Standard Python Implementation (Manual Derivative) ===")
    # Example: Find root of x^2 - 2 = 0 (Solution is sqrt(2) ≈ 1.41421356)
    
    def func(x):
        return x**2 - 2

    def d_func(x):
        return 2 * x

    x0 = 1.5
    root = newton_raphson(func, d_func, x0)
    
    if root is not None:
        print(f"Final Root: {root}")
        print(f"Actual sqrt(2): {2**0.5}")

if __name__ == "__main__":
    main()
