import jax
import jax.numpy as jnp


# ---------------------------------------------------------
# Potential (infinite square well)
# ---------------------------------------------------------
def V(x):
    return 0.0


# ---------------------------------------------------------
# PDE residual loss
# ---------------------------------------------------------
def pde_loss(pinn, x_pde):
    hbar = pinn.hbar
    m = pinn.m

    def residual(x):
        psi = pinn.psi_raw(x)
        psi_xx = pinn.psi_xx(x)
        E = pinn.energy()
        return -(hbar**2 / (2 * m)) * psi_xx + V(x) * psi - E * psi

    R_vals = jax.vmap(residual)(x_pde)
    return jnp.mean(R_vals**2)


# ---------------------------------------------------------
# Smoothness regularization
# ---------------------------------------------------------
def smoothness_loss(pinn, x_smooth):
    psi_xx_vals = pinn.psi_xx_vals(x_smooth)
    return jnp.mean(psi_xx_vals**2)


# ---------------------------------------------------------
# Energy consistency (Rayleigh quotient, RAW ψ)
# ---------------------------------------------------------
def energy_consistency_loss(pinn):
    psi = pinn.psi_raw_vals(pinn.x_norm)
    psi_x = pinn.psi_x_vals(pinn.x_norm)

    dx = pinn.dx

    kinetic = 0.5 * jnp.sum(psi_x**2) * dx
    norm = jnp.sum(psi**2) * dx
    E_rayleigh = kinetic / norm

    return (pinn.energy() - E_rayleigh)**2


# ---------------------------------------------------------
# Symmetry / parity loss (RAW ψ)
# ---------------------------------------------------------
def symmetry_loss(pinn):
    x_mid = pinn.L / 2

    if pinn.n % 2 == 0:
        return (
            pinn.psi_raw(x_mid)**2
            + antisymmetry_loss(pinn, eps=0.2)
        )
    else:
        return pinn.psi_x(x_mid)**2



# ---------------------------------------------------------
# Orthogonality loss (RAW ψ)
# ---------------------------------------------------------
def orthogonality_loss(pinn, psi_prev):
    psi = pinn.psi_raw_vals(pinn.x_norm)
    dx = pinn.dx

    # normalize both
    psi = psi / jnp.sqrt(jnp.sum(psi**2) * dx)
    psi_prev = psi_prev / jnp.sqrt(jnp.sum(psi_prev**2) * dx)

    overlap = jnp.sum(psi * psi_prev) * dx
    return overlap**2


def spectral_repulsion_loss(pinn):
    """
    Prevent intermediate eigenstates (n=2) from collapsing
    toward higher modes (n=3).
    """
    if pinn.n != 2:
        return 0.0

    x = pinn.x_norm
    L = pinn.L
    dx = pinn.dx

    # Reference higher mode (analytic ψ3 shape, unnormalized)
    psi3_ref = jnp.sin(3 * jnp.pi * x / L)

    psi = pinn.psi_raw_vals(x)

    overlap = jnp.sum(psi * psi3_ref) * dx
    return overlap**2



# ---------------------------------------------------------
# Node loss (RAW ψ)
# ---------------------------------------------------------
def node_loss(pinn):
    n = pinn.n
    L = pinn.L

    if n <= 1:
        return 0.0

    nodes = jnp.linspace(0, L, n + 1)[1:-1]
    vals = jax.vmap(pinn.psi_raw)(nodes)
    return jnp.sum(vals**2)

#antisymettrical loss
def antisymmetry_loss(pinn, eps=0.1):
    """
    Enforces ψ(L/2 + ε) = -ψ(L/2 - ε) for even n
    """
    x0 = pinn.L / 2
    return (pinn.psi_raw(x0 + eps) + pinn.psi_raw(x0 - eps))**2

def spectral_repulsion_n3(pinn):
    if pinn.n != 3:
        return 0.0

    x = pinn.x_norm
    L = pinn.L
    dx = pinn.dx

    psi1_ref = jnp.sin(jnp.pi * x / L)
    psi = pinn.psi_raw_vals(x)

    overlap = jnp.sum(psi * psi1_ref) * dx
    return overlap**2




# ---------------------------------------------------------
# Total loss
# ---------------------------------------------------------
def total_loss(
    pinn,
    x_pde,
    x_smooth,
    prev_states,
    lambdas,
):

    loss = (
        lambdas["pde"] * pde_loss(pinn, x_pde)
        + lambdas["smooth"] * smoothness_loss(pinn, x_smooth)
        + lambdas["energy"] * energy_consistency_loss(pinn)
        + lambdas["symmetry"] * symmetry_loss(pinn)
    )

    # Spectral repulsion for n=2
    loss += lambdas.get("repulsion", 0.0) * spectral_repulsion_loss(pinn)
    loss += lambdas.get("repulsion_n3", 0.0) * spectral_repulsion_n3(pinn)


    for psi_prev in prev_states.values():
        loss += lambdas["ortho"] * orthogonality_loss(pinn, psi_prev)

    if "node" in lambdas:
        loss += lambdas["node"] * node_loss(pinn)

    return loss
