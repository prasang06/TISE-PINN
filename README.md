# Physics-Informed Neural Network (PINN) Solver for the Time-Independent Schrödinger Equation

This project implements a **Physics-Informed Neural Network (PINN)** to solve the **1D Time-Independent Schrödinger Equation (TISE)** for an infinite square well potential.  
The neural network learns both the **wavefunction** and the corresponding **energy eigenvalue** by enforcing physical laws directly in the loss function, without any labeled training data.

---

## Problem Description

We consider the one-dimensional Time-Independent Schrödinger Equation:

$$
-\frac{\hbar^2}{2m}\frac{d^2 \psi(x)}{dx^2} + V(x)\psi(x) = E\psi(x)
$$

on the domain:

$$
x \in [0, L]
$$

with an infinite square well potential:

$$
V(x) =
\begin{cases}
0, & 0 < x < L \\
\infty, & \text{otherwise}
\end{cases}
$$

This imposes **Dirichlet boundary conditions**:

$$
\psi(0) = \psi(L) = 0
$$

The wavefunction must also satisfy the **normalization condition**:

$$
\int_0^L |\psi(x)|^2 \, dx = 1
$$

---

## Approach

Instead of using finite-difference or matrix diagonalization methods, this project uses a **Physics-Informed Neural Network**:

- A fully connected neural network represents the wavefunction $ \psi_\theta(x) $
- Automatic differentiation (JAX) is used to compute first and second derivatives
- The energy $E$ is treated as a **trainable parameter**
- No training data is used; the network is trained purely by enforcing physics

---

## Loss Function

The total loss is a weighted sum of three physically motivated terms.

### 1. Schrödinger Equation Residual

Enforces the TISE at collocation points inside the domain:

$$
\mathcal{L}_{PDE} =
\left\langle
\left(
-\frac{1}{2}\psi''(x) - E\psi(x)
\right)^2
\right\rangle
$$

### 2. Boundary Condition Loss

Enforces the infinite potential walls:

$$
\mathcal{L}_{BC} =
|\psi(0)|^2 + |\psi(L)|^2
$$

### 3. Normalization Loss

Ensures unit probability:

$$
\mathcal{L}_{norm} =
\left(
\int_0^L |\psi(x)|^2 dx - 1
\right)^2
$$

The full objective function is:

$$
\mathcal{L} =
\lambda_{PDE}\mathcal{L}_{PDE}
+
\lambda_{BC}\mathcal{L}_{BC}
+
\lambda_{norm}\mathcal{L}_{norm}
$$

---

## Implementation Details

- Framework: JAX
- Activation function: `tanh` (smooth second derivatives)
- Optimizer: Gradient Descent
- Automatic differentiation is used for all spatial derivatives
- Dimensionless units are used for numerical stability:

$$
\hbar = m = 1
$$

---

## Results

- The PINN successfully learns a valid eigenfunction of the infinite square well
- Boundary conditions are satisfied
- The learned energy converges to a physically meaningful eigenvalue
- The wavefunction exhibits the correct qualitative behavior for bound states

---

## Requirements

- Python 3.10+
- JAX
- NumPy
- Matplotlib
