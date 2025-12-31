import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", False)
import matplotlib.pyplot as plt
import numpy as np

# hbar = 1.054e-34     # J·s  (or set = 1 in scaled units)
# m = 9.11e-31        # kg   (electron mass, or scaled)

hbar = 1
m = 1
hidden_size = 64
seed=0

L = 10
x_pde = jnp.concatenate([
    jnp.linspace(0, L, 200),
    jnp.linspace(L*0.25, L*0.75, 400)
])

x_norm  = jnp.linspace(0, L, 150)   # normalization only
x_smooth = jnp.linspace(0, L, 150)  # smoothness only

# x_data = x_range.reshape(-1, 1)

def V(x):
    return 0.0

def init_params(seed, hidden_size):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, num=3)
    

    w1 = jax.random.normal(k1, (1,hidden_size))*0.1
    b1 = jax.random.normal(k1, (hidden_size,))*0.01

    w2 = jax.random.normal(k2, (hidden_size,hidden_size))*0.1
    b2 = jax.random.normal(k2, (hidden_size,))*0.01

    w3 = jax.random.normal(k3, (hidden_size,1))*0.1
    b3 = jax.random.normal(k3, (1,))*0.01

    E = jnp.array(0.1)

    return [w1,b1,w2,b2,w3,b3,E]

def predict(params, x):
    w1, b1, w2, b2, w3, b3, E= params
    layer1 = jnp.dot(x, w1) + b1
    layer1_activation = jnp.sin(layer1)


    layer2 = jnp.dot(layer1_activation, w2) + b2
    layer2_activation = jnp.sin(layer2)

    output = jnp.dot(layer2_activation, w3) + b3
    return output


def psi_raw(params, x): #our prediction for psi
    x_vec = jnp.array([x])
    net = predict(params, x_vec)[0]
    return x * (L - x) * net


psi_batch = jax.vmap(psi_raw, in_axes=(None, 0))
dx = L/(x_norm.shape[0] - 1)


def psi_norm_constant(params):
    psi_vals = jax.vmap(psi_raw, in_axes=(None, 0))(params, x_norm)
    norm = jnp.sqrt(jnp.sum(psi_vals**2) * dx)
    return jax.lax.stop_gradient(norm)

def psi_theta(params, x):
    psi1 = psi_raw(params, x)
    psi2 = psi_raw(params, L - x)
    return (psi1 + psi2) / (2 * psi_norm_constant(params))


psi_x = jax.grad(psi_theta, argnums=1)
psi_xx = jax.grad(psi_x, argnums=1)

def R(params, x): #residual func
    psi = psi_theta(params, x)
    psi_xx_val = psi_xx(params, x)

    E = params[-1]

    return -(hbar**2/(2*m))*psi_xx_val + V(x)*psi - E*psi

x0 = jnp.array([0.0])
xL = jnp.array([L])

params = init_params(seed, hidden_size)


R_batch = jax.vmap(R, in_axes=(None, 0))
def pde_loss(params):
    # R_vals = R_batch(params, x_data)
    return jnp.mean(R_batch(params, x_pde)**2)


lambda_pde = 50.0
lambda_bc = 100.0
lambda_norm = 10.0
lambda_smooth_loss = 5e-4
lambda_E = 1.0
lambda_sym = 5.0


def E_loss(params):
    alpha = 1e-2
    return alpha*params[-1]**2

def energy_consistency_loss(params):
    psi_vals = jax.vmap(psi_theta, in_axes=(None, 0))(params, x_norm)
    psi_x_vals = jax.vmap(psi_x, in_axes=(None, 0))(params, x_norm)

    num = 0.5 * jnp.sum(psi_x_vals**2) * dx
    den = jnp.sum(psi_vals**2) * dx
    E_rayleigh = num / den

    return (params[-1] - E_rayleigh)**2

def smoothness_loss(params):
    psi_xx_vals = jax.vmap(psi_xx, in_axes=(None, 0))(params, x_smooth)
    return jnp.mean(psi_xx_vals**2)

def symmetry_slope_loss(params):
    return psi_x(params, L/2.0)**2

def total_loss(params):
    return (lambda_pde*pde_loss(params) 
            + lambda_smooth_loss*smoothness_loss(params) 
            + lambda_E * energy_consistency_loss(params)
            + lambda_sym * symmetry_slope_loss(params)
            ) 
''' 
    + lambda_bc*boundarycond_loss(params) + lambda_norm*normal_loss(params)
    + E_loss(params) + lambda_smooth_loss*smoothness_loss(params)
    
'''
loss_grad = jax.grad(total_loss)
learning_rate = 1e-5


import optax


optimizer = optax.adam(learning_rate=1e-5)
opt_state = optimizer.init(params)

@jax.jit
def update(params, opt_state):
    loss, grads = jax.value_and_grad(total_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss



params = init_params(seed, hidden_size)

n_epochs = 15000

# for epoch in range(n_epochs):
#     params, opt_state, loss_val = update(params, opt_state)

#     if epoch % 500 == 0:
#         print(f"Epoch {epoch} | Loss={loss_val:.3e} | E={params[-1]:.6f}")


# Phase 1
optimizer = optax.adam(1e-4)
opt_state = optimizer.init(params)

for epoch in range(10000):
    params, opt_state, loss_val = update(params, opt_state)

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss={loss_val:.3e} | E={params[-1]:.6f}")

# Phase 2 (fine-tuning)
optimizer = optax.adam(1e-5)
opt_state = optimizer.init(params)

for epoch in range(5000):
    params, opt_state, loss_val = update(params, opt_state)

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss={loss_val:.3e} | E={params[-1]:.6f}")


# psi_vals = psi_batch(params, x_norm)

n=1
psi_analytical = jnp.sqrt(2/L) * jnp.sin(n*jnp.pi*x_norm/L)

psi_pinn = psi_batch(params, x_norm)
norm = jnp.sqrt(jnp.sum(psi_pinn**2) * dx)
psi_pinn = psi_pinn / norm

if jnp.dot(psi_pinn, psi_analytical) < 0:
    psi_pinn = -psi_pinn


plt.plot(x_norm, psi_pinn)
plt.plot(x_norm, psi_analytical)
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.grid()
plt.title("PINN solution of TISE")
plt.show()
