import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from CppADCodeGenEigenPy import CompiledModel


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LIB_DIR = SCRIPT_DIR + "/../lib"
DYNAMICS_MODEL_NAME = "DynamicsModel"
ROLLOUT_COST_MODEL_NAME = "RolloutCostModel"

STATE_DIM = 7 + 6
INPUT_DIM = 6


# It is convenient to create a wrapper around the C++ bindings to do things
# like build the parameter vector.
class CppDynamicsModelWrapper:
    """Wrapper around C++ bindings for DynamicsModel."""
    def __init__(self, mass, inertia):
        lib_path = os.path.join(LIB_DIR, "lib" + DYNAMICS_MODEL_NAME)
        self._model = CompiledModel(DYNAMICS_MODEL_NAME, lib_path)

        self.mass = mass
        self.inertia = inertia

    def _params(self):
        """Create the parameter vector."""
        return np.concatenate([[self.mass], self.inertia.reshape(9)])

    def evaluate(self, x, u):
        """Compute acceleration using forward dynamics."""
        inp = np.concatenate((x, u))
        return self._model.evaluate(inp, self._params())

    def jacobians(self, x, u):
        """Compute derivatives of forward dynamics w.r.t. state x and force input u."""
        inp = np.concatenate((x, u))
        J = self._model.jacobian(inp, self._params())
        dfdx = J[:, :STATE_DIM]
        dfdu = J[:, STATE_DIM:]
        return dfdx, dfdu


class JaxDynamicsModel:
    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = jnp.linalg.inv(inertia)

        self.dfdx = jax.jit(jax.jacfwd(self.evaluate, argnums=0))
        self.dfdu = jax.jit(jax.jacfwd(self.evaluate, argnums=1))

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, x, u):
        force, torque = u[:3], u[3:]
        v = x[7:10]
        omega = x[10:]

        a = force / self.mass
        alpha = self.inertia_inv @ (torque - jnp.cross(omega, self.inertia @ omega))

        return jnp.concatenate((a, alpha))

    def jacobians(self, x, u):
        return self.dfdx(x, u), self.dfdu(x, u)


def main():
    # model parameters
    mass = 1.0
    inertia = np.eye(3)

    # state and input
    x = np.zeros(STATE_DIM)
    x[6] = 1  # for quaternion
    u = np.ones(INPUT_DIM)

    # C++-based model which we bind to and load from a shared lib
    cpp_model = CppDynamicsModelWrapper(mass, inertia)
    A = cpp_model.evaluate(x, u)
    dfdx, dfdu = cpp_model.jacobians(x, u)

    # jax-based model which is computed just in time
    jax_model = JaxDynamicsModel(mass, inertia)
    A_jax = jax_model.evaluate(x, u)
    dfdx_jax, dfdu_jax = jax_model.jacobians(x, u)


if __name__ == "__main__":
    main()
