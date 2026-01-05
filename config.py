import jax.numpy as jnp

L = 10.0
hbar = 1.0
m = 1.0

hidden_size = 128
seed = 0

x_pde = jnp.concatenate([
    jnp.linspace(0, L, 300),
    jnp.linspace(0.2 * L, 0.8 * L, 600),
])

x_norm = jnp.linspace(0.0, L, 150)

x_smooth = jnp.linspace(0.0, L, 150)

dx = L / (x_norm.shape[0] - 1)
