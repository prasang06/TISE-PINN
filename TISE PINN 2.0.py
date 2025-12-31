import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

# hbar = 1.054e-34     # J·s  (or set = 1 in scaled units)
# m = 9.11e-31        # kg   (electron mass, or scaled)

hbar = 1
m = 1

L = 10
x_range = jnp.linspace(0, L, 1000)
x_data = x_range.reshape(-1, 1)

V_walls = 1e8
condition = (x_range>0) & (x_range<L)
V_values = jnp.where(condition, 0.0, V_walls)

def V(x):
    return 0.0

V_data = V_values.reshape(-1, 1)


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
    layer1_activation = jnp.tanh(layer1)


    layer2 = jnp.dot(layer1_activation, w2) + b2
    layer2_activation = jnp.tanh(layer2)

    output = jnp.dot(layer2_activation, w3) + b3
    return output


def psi_raw(params, x): #our prediction for psi
    x_vec = jnp.array([x])
    net = predict(params, x_vec)[0]
    return x * (L - x) * net


psi_batch = jax.vmap(psi_raw, in_axes=(None, 0))
dx = L/(x_range.shape[0] - 1)


def psi_norm_constant(params):
    psi_vals = jax.vmap(psi_raw, in_axes=(None, 0))(params, x_range)
    norm = jnp.sqrt(jnp.sum(psi_vals**2) * dx)
    return jax.lax.stop_gradient(norm)

def psi_theta(params, x):
    psi1 = psi_raw(params, x)
    psi2 = psi_raw(params, L - x)
    return (psi1 + psi2) / (2 * psi_norm_constant(params))


psi_x = jax.grad(psi_theta, argnums=1)
psi_xx = jax.grad(psi_x, argnums=1)

def R(params, x): #residual func
    x=x[0]
    psi = psi_theta(params, x)
    psi_xx_val = psi_xx(params, x)

    E = params[-1]

    return -(hbar**2/(2*m))*psi_xx_val + V(x)*psi - E*psi

x0 = jnp.array([0.0])
xL = jnp.array([L])

# def boundarycond_loss(params):
#     psi_0 = psi_theta(params, 0.0)
#     psi_L = psi_theta(params, L)

#     return psi_0**2 + psi_L**2

params = init_params(0, 32)

# print(psi_theta(params, jnp.array([0.0])))
# print(psi_theta(params, jnp.array([L])))
# print(boundarycond_loss(params))

# def normal_loss(params):
#     psi_vals = psi_batch(params, x_data)
#     norm = jnp.sum(psi_vals**2)*dx
#     return (norm-1)**2


# params = init_params(0, 32)
# print(normal_loss(params))


R_batch = jax.vmap(R, in_axes=(None, 0))
def pde_loss(params):
    # R_vals = R_batch(params, x_data)
    return jnp.mean(R_batch(params, x_data)**2)


lambda_pde = 10.0
lambda_bc = 100.0
lambda_norm = 10.0
lambda_smooth_loss = 1e-2

def E_loss(params):
    alpha = 1e-2
    return alpha*params[-1]**2

def smoothness_loss(params):
    psi_xx_vals = jax.vmap(psi_xx, in_axes=(None, 0))(params, x_range)
    return jnp.mean(psi_xx_vals**2)


def total_loss(params):
    return lambda_pde*pde_loss(params) + E_loss(params) + lambda_smooth_loss*smoothness_loss(params) # + lambda_bc*boundarycond_loss(params)  + lambda_norm*normal_loss(params)

loss_grad = jax.grad(total_loss)
learning_rate = 1e-4


import optax


optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

@jax.jit
def update(params, opt_state):
    loss, grads = jax.value_and_grad(total_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss



params = init_params(seed=0, hidden_size=32)

n_epochs = 20000

# for epoch in range(n_epochs):
#     params = update(params, learning_rate)

#     if epoch % 500 == 0:
#         loss_val = total_loss(params)
#         E_val = params[-1]
#         print(f"Epoch {epoch} | Loss = {loss_val:.6e} | E = {E_val:.6e}")

for epoch in range(n_epochs):
    params, opt_state, loss_val = update(params, opt_state)

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss={loss_val:.3e} | E={params[-1]:.6f}")

psi_vals = psi_batch(params, x_data)

plt.plot(x_range, psi_vals)
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.grid()
plt.title("PINN solution of TISE")
plt.show()
