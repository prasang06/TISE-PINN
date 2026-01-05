import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from config import (
    L, hidden_size, seed,
    x_pde, x_norm, x_smooth, dx
)
from model import EigenPINN
from trainer import SpectrumTrainer

lambdas = {
    "pde": 5.0,
    "smooth": 1e-3,
    "energy": 1.0,
    "symmetry": 10.0,
    "ortho": 200.0,
    "node": 20.0,
}



trainer = SpectrumTrainer(
    x_pde=x_pde,
    x_smooth=x_smooth,
    lambdas=lambdas,
    learning_rate=1e-4,
    fine_tune_lr=1e-5,
    epochs=25000,
    fine_tune_epochs=5000,
)

def fix_sign(psi, x, n, L):
    ref = jnp.sin(n * jnp.pi * x / L)
    if jnp.dot(psi, ref) < 0:
        return -psi
    return psi


def make_pinn(n):
    return EigenPINN(
        n=n,
        L=L,
        hidden_size=hidden_size,
        seed=seed + 100 * n,
        x_norm=x_norm,
        dx=dx,
    )


n_max = 3
trained = trainer.solve_spectrum(make_pinn, n_max=n_max)

# Plot results
plt.figure(figsize=(8, 5))

for n, pinn in trained.items():

    # RAW wavefunction
    psi = pinn.psi_raw_vals(x_norm)

    # Normalize for plotting
    psi = psi / jnp.sqrt(jnp.sum(psi**2) * dx)

    # Fix global sign (important!)
    psi = fix_sign(psi, x_norm, n, L)

    plt.plot(x_norm, psi, label=f"PINN n={n}")

    # Exact solution
    psi_exact = jnp.sqrt(2 / L) * jnp.sin(n * jnp.pi * x_norm / L)
    plt.plot(x_norm, psi_exact, "--", label=f"Exact n={n}")

plt.xlabel("x")
plt.ylabel(r"$\psi_n(x)$")
plt.legend()
plt.title("Eigenstates of 1D Infinite Square Well (PINN)")
plt.tight_layout()
plt.show()

# Diagnostics: overlaps + energies
print("\n=== Diagnostics ===")

for n, pinn in trained.items():
    print(f"n={n} | E_PINN = {pinn.energy():.6f} "
          f"| E_exact = {(n * jnp.pi / L)**2 / 2:.6f}")

print("\nOverlaps ⟨ψ_n | ψ_m⟩:")
for i in range(1, n_max + 1):
    for j in range(1, n_max + 1):
        psi_i = trainer.eigenstates[i]
        psi_j = trainer.eigenstates[j]
        overlap = jnp.sum(psi_i * psi_j) * dx
        print(f"<{i}|{j}> = {overlap:.3e}", end="   ")
    print()

print("\nNode counts:")
for n, psi in trainer.eigenstates.items():
    nodes = jnp.sum(jnp.diff(jnp.sign(psi)) != 0)
    print(f"n={n} → nodes ≈ {nodes}")
