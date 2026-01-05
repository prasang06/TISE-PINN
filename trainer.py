import jax
import jax.numpy as jnp
import optax
from functools import partial

from losses import total_loss


class SpectrumTrainer:

    def __init__(
        self,
        x_pde,
        x_smooth,
        lambdas,
        learning_rate=1e-4,
        fine_tune_lr=3e-6,
        epochs=15000,
        fine_tune_epochs=5000,
    ):
        self.x_pde = x_pde
        self.x_smooth = x_smooth
        self.lambdas = lambdas

        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs

        # stores ψ_n(x_norm)
        self.eigenstates = {}

    # ---------------------------------------------------------
    # JIT-safe update step (PARAMS ONLY)
    # ---------------------------------------------------------
    @staticmethod
    @partial(jax.jit, static_argnames=("pinn_static", "optimizer_update"))

    def _update_step(
        params,
        opt_state,
        optimizer_update,
        x_pde,
        x_smooth,
        prev_states,
        lambdas,
        pinn_static,
    ):
        """
        One JAX-safe optimizer step
        """

        def loss_fn(params):
            pinn_static.params = params
            return total_loss(
                pinn_static,
                x_pde=x_pde,
                x_smooth=x_smooth,
                prev_states=prev_states,
                lambdas=lambdas,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer_update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss
    # ---------------------------------------------------------
    # Train ONE eigenstate
    # ---------------------------------------------------------
    def train_state(self, pinn, verbose=True):

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(pinn.params)

        for epoch in range(self.epochs):

            if pinn.n >= 3:
                pde_scale = min(1.0, epoch / 8000)
            else:
                pde_scale = min(1.0, epoch / 4000)


            lambdas_dynamic = dict(self.lambdas)
            lambdas_dynamic["pde"] = pde_scale * self.lambdas["pde"]
            # Reduce energy stiffness for higher modes
            lambdas_dynamic["energy"] /= pinn.n


            # --- Spectral repulsion (stage-aware, one-way) ---
            if pinn.n == 2:
                lambdas_dynamic["repulsion"] = 50.0
            else:
                lambdas_dynamic["repulsion"] = 0.0

            if pinn.n == 3:
                lambdas_dynamic["repulsion_n3"] = 30.0
            else:
                lambdas_dynamic["repulsion_n3"] = 0.0


            if pinn.n % 2 == 0:
                lambdas_dynamic["node"] *= 1.5
                

            params, opt_state, loss = self._update_step(
                pinn.params,
                opt_state,
                optimizer.update,
                self.x_pde,
                self.x_smooth,
                self.eigenstates,
                lambdas_dynamic,
                pinn,
            )

            pinn.params = params

            if verbose and epoch % 500 == 0:
                print(
                    f"[n={pinn.n}] "
                    f"Epoch {epoch:6d} | "
                    f"Loss = {loss:.3e} | "
                    f"E = {pinn.energy():.6f}"
                )

        # -------- Fine tuning --------
        optimizer = optax.adam(self.fine_tune_lr)
        opt_state = optimizer.init(pinn.params)

        for _ in range(self.fine_tune_epochs):
            params, opt_state, _ = self._update_step(
                pinn.params,
                opt_state,
                optimizer.update,
                self.x_pde,
                self.x_smooth,
                self.eigenstates,
                lambdas_dynamic,
                pinn,
            )
            pinn.params = params

        psi_raw = pinn.psi_raw_vals(pinn.x_norm)
        self.eigenstates[pinn.n] = jax.lax.stop_gradient(psi_raw)

        return pinn

    # ---------------------------------------------------------
    # Train spectrum up to n_max
    # ---------------------------------------------------------
    def solve_spectrum(self, pinn_factory, n_max, verbose=True):

        trained = {}

        for n in range(1, n_max + 1):
            if verbose:
                print(f"\n=== Training eigenstate n={n} ===")

            pinn = pinn_factory(n)
            pinn = self.train_state(pinn, verbose=verbose)

            trained[n] = pinn

        return trained
