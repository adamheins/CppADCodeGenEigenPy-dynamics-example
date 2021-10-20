import numpy as np
import os

from CppADCodeGenEigenPy import CompiledModel

import IPython


LIB_DIR = "lib"
DYNAMICS_MODEL_NAME = "DynamicsModel"
ROLLOUT_COST_MODEL_NAME = "RolloutCostModel"

STATE_DIM = 7 + 6
INPUT_DIM = 6


# It is convenient to create a wrapper around the C++ bindings to do things
# like build the parameter vector.
class DynamicsModel:
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


def main():
    dynamics_model = DynamicsModel(1, np.eye(3))
    # rollout_cost_model = load_model(ROLLOUT_COST_MODEL_NAME)
    x = np.zeros(STATE_DIM)
    x[6] = 1  # for quaternion
    u = np.ones(INPUT_DIM)

    # compute acceleration
    A = dynamics_model.evaluate(x, u)
    dfdx, dfdu = dynamics_model.jacobians(x, u)

    IPython.embed()


if __name__ == "__main__":
    main()
