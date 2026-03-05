# Automatic Differentiation (AD) Simply Explained

Automatic Differentiation (often abbreviated as AD) is a technique for **exactly** computing the derivatives of functions. It is neither symbolic differentiation nor numerical differentiation, but a "third way" between the two. Below, I will explain its mathematical principles in the simplest possible way.

---

## 1. Distinguishing Three Differentiation Methods

Suppose we want to differentiate the function $f(x) = x^2 \sin(x)$.

**Symbolic Differentiation** is what you do in calculus class: manually applying differentiation rules to get $f'(x) = 2x\sin(x) + x^2\cos(x)$. It is exact, but the resulting expression can become exponentially large.

**Numerical Differentiation** uses finite difference approximation: $f'(x) \approx \frac{f(x+h) - f(x)}{h}$. It is simple to implement but suffers from rounding errors and truncation errors.

**Automatic Differentiation** is based on the core idea that: no matter how complex a function is, when a computer executes it, it is a composition of a sequence of **elementary operations** (addition, subtraction, multiplication, division, sin, cos, exp...). We only need to know the derivatives of these elementary operations and then **chain them together using the chain rule** to obtain the exact derivative value.

---

## 2. Core Mathematical Foundation: The Chain Rule

The chain rule is the soul of automatic differentiation. Recall:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

If a computation process is nested layer by layer:

$$
x \xrightarrow{f_1} v_1 \xrightarrow{f_2} v_2 \xrightarrow{f_3} \cdots \xrightarrow{f_n} y
$$

Then:

$$
\frac{dy}{dx} = \frac{dy}{dv_{n-1}} \cdot \cdots \cdot \frac{dv_3}{dv_2} \cdot \frac{dv_2}{dv_1} \cdot \frac{dv_1}{dx}
$$

The key question is: **From which end do we start multiplying?** This leads to two different modes.

---

## 3. Forward Mode

**Proceeds from the input end towards the output end.**

For example, let's compute the derivative of $f(x) = \sin(x^2)$ at $x=3$:

Decompose the calculation into elementary steps:

| Step | Value (primal) | Derivative (tangent) |
|---|---|---|
| $v_0 = x$ | $v_0 = 3$ | $\dot{v}_0 = 1$ (derivative w.r.t $x$, seed value) |
| $v_1 = v_0^2$ | $v_1 = 9$ | $\dot{v}_1 = 2v_0 \cdot \dot{v}_0 = 6$ |
| $v_2 = \sin(v_1)$ | $v_2 = \sin(9)$ | $\dot{v}_2 = \cos(v_1) \cdot \dot{v}_1 = 6 \cos(9)$ |

Notice that at each step, we compute both the **value** and the **derivative** simultaneously, moving forward together like "twin brothers". The final $\dot{v}_2$ is $f'(3)$.

**Mathematical Essence: Dual Numbers — The "Magic" of AD**

Mathematically, the process of forward mode is completely equivalent to performing operations on **Dual Numbers**. This might sound advanced, but it is very similar to Complex Numbers.

1.  **Definition**:
    Complex numbers have an imaginary unit $i$ satisfying $i^2 = -1$.
    Dual numbers define a similar unit $\varepsilon$ satisfying $\varepsilon^2 = 0$ (but $\varepsilon \neq 0$).
    A dual number can be written as $a + b\varepsilon$, where $a$ is the **real part** (Primal, corresponding to the function value) and $b$ is the **dual part** (Tangent, corresponding to the derivative value).

2.  **Arithmetic Rules**:
    Using the property $\varepsilon^2 = 0$, operations on dual numbers become very simple:
    *   **Addition**: $(a + b\varepsilon) + (c + d\varepsilon) = (a+c) + (b+d)\varepsilon$
    *   **Multiplication**: $(a + b\varepsilon)(c + d\varepsilon) = ac + ad\varepsilon + bc\varepsilon + bd\varepsilon^2 = ac + (ad+bc)\varepsilon$
        *   Look! $ac$ is the product of the original function values, and $(ad+bc)$ is exactly the form of the product rule $(uv)' = u'v + uv'$!

3.  **Why does this compute derivatives? Taylor Expansion**:
    The Taylor expansion of any differentiable function $f(x)$ at $x$ is:
    $$f(x + h) = f(x) + f'(x)h + \frac{f''(x)}{2!}h^2 + \cdots$$
    If we let $h = \varepsilon$ (the dual unit), since $\varepsilon^2 = 0, \varepsilon^3=0 \dots$, all higher-order terms vanish! We are left with:
    $$f(x + \varepsilon) = f(x) + f'(x)\varepsilon$$
    This is simply magic: **Just replace the input $x$ with $x + 1\varepsilon$ and evaluate the function. The real part of the result is the function value $f(x)$, and the dual part (coefficient of $\varepsilon$) is the derivative value $f'(x)$!**

4.  **Practical Walkthrough**:
    Let's re-calculate $f(x) = \sin(x^2)$ at $x=3$ for value and derivative.
    We need to convert the input $x=3$ into a dual number.
    Let $x = 3 + 1\varepsilon$.
    *   **Real part 3**: Represents the current **value** of $x$.
    *   **Dual part 1**: Represents the **derivative seed** of $x$. Since we are computing the derivative of $f(x)$ with respect to $x$, and $\frac{dx}{dx}=1$, the dual part is set to 1.
    *   **Step 1: $x^2$**:
        $$(3 + \varepsilon)^2 = 3^2 + 2\cdot 3 \cdot \varepsilon + \varepsilon^2 = 9 + 6\varepsilon$$
        (Value is 9, derivative is 6)
    *   **Step 2: $\sin(\cdot)$**:
        $$\sin(9 + 6\varepsilon)$$
        Using the formula $\sin(A+B) = \sin A \cos B + \cos A \sin B$, and since $\varepsilon$ is very small, $\cos(k\varepsilon) \approx 1, \sin(k\varepsilon) \approx k\varepsilon$ (or simply applying $f(x+\Delta) = f(x) + f'(x)\Delta$):
        $$\sin(9 + 6\varepsilon) = \sin(9) + \cos(9) \cdot 6\varepsilon$$
    *   **Result**:
        The real part $\sin(9)$ is the function value, and the dual part $6\cos(9)$ is exactly the derivative!

This explains why in forward mode, we only need to overload elementary operators to automatically compute derivatives.

---

## 4. Reverse Mode

**Proceeds from the output end towards the input end — this is the famous "Backpropagation" in Deep Learning.**

Same example $f(x) = \sin(x^2)$:

**Step 1: Forward Pass**, record the entire computation graph (but do not compute derivatives).

**Step 2: Backward Pass (Propagate derivatives)**. Define the "adjoint" $\bar{v}_i = \frac{\partial y}{\partial v_i}$:

| Backward Step | Adjoint Value |
|---|---|
| $\bar{v}_2 = \frac{\partial y}{\partial v_2}$ | $= 1$ (Output w.r.t itself) |
| $\bar{v}_1 = \bar{v}_2 \cdot \cos(v_1)$ | $= \cos(9)$ |
| $\bar{v}_0 = \bar{v}_1 \cdot 2v_0$ | $= 6 \cos(9)$ |

The result is exactly the same as in forward mode!

---

## 5. Trade-offs Between the Two Modes

The core difference lies in **efficiency**:

Suppose the function is $f: \mathbb{R}^n \to \mathbb{R}^m$.

**Forward Mode** can compute all partial derivatives with respect to **one input variable** in a single pass. If there are $n$ input variables, you need to run it $n$ times. So it is suitable when $n$ is small and $m$ is large.

**Reverse Mode** can compute partial derivatives of the output with respect to **all input variables** in a single pass. If there are $m$ outputs, you need to run it $m$ times. So it is suitable when $n$ is large and $m$ is small.

In Deep Learning, the loss function is a scalar ($m=1$), while the parameters can number in the millions ($n$ is huge), so Reverse Mode (Backpropagation) is naturally the best choice — **a single backward pass gives gradients for all parameters**.

---

## 6. One-Sentence Summary

The essence of Automatic Differentiation is: **Decompose a complex function into a computation graph of elementary operations, use the known analytical derivatives of each elementary operation, and automatically combine them via the chain rule to obtain the exact overall derivative.** It possesses both the precision of symbolic differentiation and the efficiency of numerical computation.
