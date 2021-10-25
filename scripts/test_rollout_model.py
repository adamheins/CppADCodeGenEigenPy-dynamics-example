import os
from functools import partial
import time
import timeit

import numpy as np
import jax
import jax.numpy as jnp

from CppADCodeGenEigenPy import CompiledModel


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LIB_DIR = SCRIPT_DIR + "/../lib"
ROLLOUT_COST_MODEL_NAME = "RolloutCostModel"

STATE_DIM = 7 + 6
INPUT_DIM = 6

# NOTE these are currently fixed on the C++ side
NUM_TIME_STEPS = 1
TIMESTEP = 0.1


class CppRolloutCostModelWrapper:
    """Wrapper around C++ bindings for DynamicsModel."""

    def __init__(self, mass, inertia):
        lib_path = os.path.join(LIB_DIR, "lib" + ROLLOUT_COST_MODEL_NAME)
        self._model = CompiledModel(ROLLOUT_COST_MODEL_NAME, lib_path)

        self.mass = mass
        self.inertia = inertia

    def _params(self, xds):
        """Create the parameter vector."""
        return np.concatenate([[self.mass], self.inertia.flatten(), xds.flatten()])

    def _input(self, x0, us):
        return np.concatenate((x0, us.flatten()))

    def evaluate(self, x0, us, xds):
        """Compute acceleration using forward dynamics."""
        return self._model.evaluate(self._input(x0, us), self._params(xds))

    def jacobians(self, x0, us, xds):
        """Compute derivatives of forward dynamics w.r.t. state x and force input u."""
        J = self._model.jacobian(self._input(x0, us), self._params(xds))
        dfdx0 = J[:, :STATE_DIM]
        dfdus = J[:, STATE_DIM:]
        return dfdx0, dfdus


def decompose_state(x):
    """Decompose state into position, orientation, linear and angular velocity."""
    r = x[:3]
    q = x[3:7]
    v = x[7:10]
    ω = x[10:]
    return r, q, v, ω


def orientation_error(q, qd):
    """Error between two quaternions."""
    # This is the vector portion of qd.inverse() * q
    return qd[3] * q[:3] - q[3] * qd[:3] - jnp.cross(qd[:3], q[:3])


def state_error(x, xd):
    r, q, v, ω = decompose_state(x)
    rd, qd, vd, ωd = decompose_state(xd)

    r_err = rd - r
    q_err = orientation_error(q, qd)
    v_err = vd - v
    ω_err = ωd - ω

    return jnp.concatenate((r_err, q_err, v_err, ω_err))


def quaternion_multiply(q0, q1):
    v0, w0 = q0[:3], q0[3]
    v1, w1 = q1[:3], q1[3]
    return jnp.append(w0 * v1 + w1 * v0 + jnp.cross(v0, v1), w0 * w1 - v0 @ v1)


def integrate_state(x0, A, dt):
    r0, q0, v0, ω0 = decompose_state(x0)
    a, α = A[:3], A[3:]

    r1 = r0 + dt * v0 + 0.5 * dt ** 2 * a
    v1 = v0 + dt * a

    ω1 = ω0 + dt * α

    aa = 0.5 * dt * (ω0 + ω1)
    angle = jnp.linalg.norm(aa)
    axis = aa / angle

    c = jnp.cos(0.5 * angle)
    s = jnp.sin(0.5 * angle)
    qw = jnp.append(s * axis, c)
    q1 = quaternion_multiply(qw, q0)

    return jnp.concatenate((r1, q1, v1, ω1))


class JaxRolloutCostModel:
    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = jnp.linalg.inv(inertia)

        self.dfdx0 = jax.jit(jax.jacfwd(self.evaluate, argnums=0))
        self.dfdus = jax.jit(jax.jacfwd(self.evaluate, argnums=1))

    @partial(jax.jit, static_argnums=(0,))
    def forward_dynamics(self, x, u):
        f, τ = u[:3], u[3:]
        _, _, v, ω = decompose_state(x)

        a = f / self.mass
        α = self.inertia_inv @ (τ - jnp.cross(ω, self.inertia @ ω))

        return jnp.concatenate((a, α))

    def evaluate(self, x0, us, xds):
        def state_func(x, u):
            A = self.forward_dynamics(x, u)
            x = integrate_state(x, A, TIMESTEP)
            return x, x

        _, xs = jax.lax.scan(state_func, x0, us)

        def cost_func(cost, datum):
            x, xd, u = (
                datum[:STATE_DIM],
                datum[STATE_DIM : 2 * STATE_DIM],
                datum[-INPUT_DIM:],
            )
            e = state_error(x, xd)
            cost = cost + 0.5 * (e @ e + 0.1 * u @ u)
            return cost, datum

        data = jnp.hstack((xs, xds, us))
        cost, _ = jax.lax.scan(cost_func, 0, data)

        return cost

    def jacobians(self, x0, us, xds):
        return self.dfdx0(x0, us, xds), self.dfdus(x0, us, xds)


def zero_state():
    x = np.zeros(STATE_DIM)
    x[6] = 1  # for quaternion
    return x


def main():
    np.random.seed(0)

    # model parameters
    mass = 1.0
    inertia = np.eye(3)

    # initial state
    x0 = zero_state()

    # force/torque inputs
    us = np.random.random((NUM_TIME_STEPS, INPUT_DIM))

    # desired states
    xd = zero_state()
    xd[:3] = [1, 1, 1]  # want body to move position
    xds = np.tile(xd, (NUM_TIME_STEPS, 1))

    # C++-based model which we bind to and load from a shared lib
    t = time.time()
    cpp_model = CppRolloutCostModelWrapper(mass, inertia)
    dfdx0_cpp, dfdus_cpp = cpp_model.jacobians(x0, us, xds)
    print(f"Time to load C++ model jacobians = {time.time() - t} sec")

    # jax-based model which is computed just in time
    t = time.time()
    jax_model = JaxRolloutCostModel(mass, inertia)
    dfdx0_jax, dfdus_jax = jax_model.jacobians(x0, us, xds)
    print(f"Time to load JAX model jacobians = {time.time() - t} sec")

    cost_cpp = cpp_model.evaluate(x0, us, xds)
    cost_jax = jax_model.evaluate(x0, us, xds)

    # check that both models actually get the same results
    assert np.isclose(cost_cpp, cost_jax), "Cost is not the same between models."
    assert np.isclose(dfdx0_cpp, dfdx0_jax).all(), "dfdx0 is not the same between models."
    assert np.isclose(dfdus_cpp, dfdus_jax).all(), "dfdus is not the same between models."

    # compare runtime evaluation time
    n = 100000
    cpp_time = timeit.timeit(
        "cpp_model.jacobians(x0, us, xds)", number=n, globals=locals()
    )
    jax_time = timeit.timeit(
        "jax_model.jacobians(x0, us, xds)", number=n, globals=locals()
    )
    print(f"Time to evaluate C++ model jacobians {n} times = {cpp_time} sec")
    print(f"Time to evaluate JAX model jacobians {n} times = {jax_time} sec")


if __name__ == "__main__":
    main()
