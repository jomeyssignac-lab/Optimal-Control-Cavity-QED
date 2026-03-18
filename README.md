# Optimal-Control-Cavity-QED
import numpy as np
import matplotlib.pyplot as plt
import qutip
from qutip import basis, sigmay, sigmaz, mesolve, Bloch

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from scipy.optimize import minimize

# Config JAX (Toujours CPU pour Windows)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# ==========================================
# 1. Setting Target State
# ==========================================
# Theta : Altitude (0 = Haut/Nord, pi = Bas/Sud, pi/2 = Équateur)
# Phi   : Azimut   (Rotation autour de l'axe Z)

THETA = np.pi / 2   
PHI   = 0.0         

# Formula : cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>
target_state_qobj = (np.cos(THETA/2) * basis(2,0) + 
                     np.exp(1j * PHI) * np.sin(THETA/2) * basis(2,1))
# ==========================================
# 2. Control Parameters
# ==========================================
T = 3.0          
n_ts = 800       
dt = T / n_ts

# Hamiltonian : H = (wa/2)*Sz + u(t)*Sy
wa = 2 * np.pi   # Frequency of qubit
H_drift_jax = jnp.array((wa / 2.0) * sigmaz().full(), dtype=jnp.complex128)
H_control_jax = jnp.array(sigmay().full(), dtype=jnp.complex128)

# JAX states
psi0_jax = jnp.array(basis(2, 0).full(), dtype=jnp.complex128)
psi_targ_jax = jnp.array(target_state_qobj.full(), dtype=jnp.complex128)

n_gauss = 1  # Number of Gaussians
x0 = np.array([1.0, T/2.0, T/4.0]) # [Height, Center, Width]

def pulse_func(t, params):
    val = 0.0
    for i in range(0, len(params), 3):
        val += params[i] * jnp.exp(-(t - params[i+1])**2 / (2 * params[i+2]**2)) * jnp.cos(wa*t)
    return val

@jax.jit
def fidelity_loss(params):
    def step_scan(psi, i):
        t = i * dt
        H_t = H_drift_jax + pulse_func(t, params) * H_control_jax
        U = expm(-1j * H_t * dt)
        return jnp.dot(U, psi), None
    indices = jnp.arange(n_ts)
    final_psi, _ = jax.lax.scan(step_scan, psi0_jax, indices)
    overlap = jnp.dot(psi_targ_jax.conj().T, final_psi)
    return 1.0 - jnp.abs(jnp.squeeze(overlap))**2

# Optimisation
loss_and_grad = jax.value_and_grad(fidelity_loss)
def func_scipy(x):
    v, g = loss_and_grad(jnp.array(x))
    return np.array(v, dtype=np.float64), np.array(g, dtype=np.float64)

res = minimize(func_scipy, x0, method='BFGS', jac=True)
print(f" Infidelity : {res.fun:.8f}")

tlist = np.linspace(0, T, n_ts)
pulse_vals = [pulse_func(t, res.x) for t in tlist]

# Area below pulse curve
aire_pulse = np.trapezoid(pulse_vals, tlist)
print(f" Area of Pulse (Theory: Pi=3.14) : {aire_pulse:.4f}")

# Simulation 
sim = mesolve([(wa/2.0)*sigmaz(), [sigmay(), np.array(pulse_vals)]], basis(2,0), tlist, [], [])

psi_final = sim.states[-1]
recouvrement = target_state_qobj.overlap(psi_final)
fidelite = np.abs(recouvrement)**2
print(f" fidelity according to (QuTiP) : {fidelite:.10f}")
pop_1 = qutip.expect(qutip.num(2), sim.states)



#Matplotlib and Bloch sphere

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(tlist, pulse_vals, 'g-')
plt.title(f"Simple Pulse (Area ~ {aire_pulse:.2f})")
plt.fill_between(tlist, pulse_vals, color='green', alpha=0.1) # Joli remplissage
#plt.savefig("simple_realistic_pulse.pdf", format="pdf", bbox_inches="tight")

plt.subplot(1, 2, 2)
plt.plot(tlist, pop_1, 'r-')
plt.axhline(np.sin(THETA/2)**2, c='k', ls=':')
plt.title("Qubit population")
plt.savefig("simple_realistic_area_pulse.pdf", format="pdf", bbox_inches="tight")
plt.show()

b = Bloch()

# add_states not working
expect_x = qutip.expect(qutip.sigmax(), sim.states)
expect_y = qutip.expect(qutip.sigmay(), sim.states)
expect_z = qutip.expect(qutip.sigmaz(), sim.states)

b.add_points([expect_x, expect_y, expect_z], meth='s')

vec_final = [expect_x[-1], expect_y[-1], expect_z[-1]]
b.add_vectors(vec_final)
b.save("optimized_bloch_trajectory.pdf")
b.show()
print(f"📍 Coordonnées finales de la flèche :")
print(f"   X = {expect_x[-1]:.4f}  (On veut 1.0000)")
print(f"   Y = {expect_y[-1]:.4f}  (On veut 0.0000)")
print(f"   Z = {expect_z[-1]:.4f}  (On veut 0.0000)")
