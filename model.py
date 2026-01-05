import jax
import jax.numpy as jnp
from functools import partial

class EigenPINN:
    def __init__(self, n, L, hidden_size, seed,
                 hbar=1.0, m=1.0,
                 x_norm=None, dx=None):

        self.n = n
        self.L = L
        self.hbar = hbar
        self.m = m

        self.x_norm = x_norm
        self.dx = dx

        self.params = self._init_params(seed, hidden_size)


    def _init_params(self, seed, hidden_size):
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)

        w1 = jax.random.normal(k1, (1, hidden_size)) * 0.1
        b1 = jax.random.normal(k1, (hidden_size,)) * 0.01

        w2 = jax.random.normal(k2, (hidden_size, hidden_size)) * 0.1
        b2 = jax.random.normal(k2, (hidden_size,)) * 0.01

        w3 = jax.random.normal(k3, (hidden_size, 1)) * 0.1
        b3 = jax.random.normal(k3, (1,)) * 0.01

        # energy parameter (initialized near analytic value)
        E0 = (self.n * jnp.pi / self.L)**2 / 2
        E = jnp.array(E0)

        return [w1, b1, w2, b2, w3, b3, E]

    def _forward(self, x):
        w1, b1, w2, b2, w3, b3, _ = self.params

        x = x.reshape(-1, 1)

        z1 = jnp.dot(x, w1) + b1
        a1 = jnp.sin(z1)

        z2 = jnp.dot(a1, w2) + b2
        a2 = jnp.sin(z2)

        out = jnp.dot(a2, w3) + b3
        return out.squeeze()

    def psi_raw(self, x):
        net = self._forward(jnp.array(x))
        return x * (self.L - x) * net

    def psi_raw_vals(self, x):
        return jax.vmap(self.psi_raw)(x)

    def psi_vals(self):
        psi = self.psi_raw_vals(self.x_norm)
        norm = jnp.sqrt(jnp.sum(psi**2) * self.dx)
        return psi / jax.lax.stop_gradient(norm)

    def psi_x(self, x):
        return jax.grad(self.psi_raw)(x)

    def psi_xx(self, x):
        return jax.grad(self.psi_x)(x)

    def psi_x_vals(self, x):
        return jax.vmap(self.psi_x)(x)

    def psi_xx_vals(self, x):
        return jax.vmap(self.psi_xx)(x)

    def energy(self):
        return self.params[-1]
