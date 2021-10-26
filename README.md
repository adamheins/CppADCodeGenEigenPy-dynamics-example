# CppADCodeGenEigenPy Dynamics Example

This repo shows an example of using
[CppADCodeGenEigenPy](https://github.com/adamheins/CppADCodeGenEigenPy) to
differentiate functions related to rigid body dynamics.

The
[DynamicsModel](https://github.com/adamheins/CppADCodeGenEigenPy-dynamics-example/blob/main/include/dynamics_model.h)
shows how to differentiate basic forward dynamics computed using the
Newton-Euler equations of motion. The
[RolloutCostModel](https://github.com/adamheins/CppADCodeGenEigenPy-dynamics-example/blob/main/include/rollout_model.h)
shows how to differentiate a scalar cost function of the system state and input
forward simulated over a time horizon, similar to what may be used for model
predictive control.

There are Python
[scripts](https://github.com/adamheins/CppADCodeGenEigenPy-dynamics-example/tree/main/scripts)
that:
1. show how to use the auto-differentiated models from Python, and
2. compare the performance (both startup and runtime) with
   [JAX](https://github.com/google/jax), a popular Python autodiff library.

## Installation and usage

First, follow the
[instructions](https://github.com/adamheins/CppADCodeGenEigenPy) to install
CppADCodeGenEigenPy.

Next, get a copy of this repo:
```
git clone https://github.com/adamheins/CppADCodeGenEigenPy-dynamics-example.git
cd CppADCodeGenEigenPy-dynamics-example
```

Compile the auto-differentiated models:
```
# this compiles the program to generate the model
make compiler

# use the program compiled by the above line to actually produce the model
# (which is shared object .so file)
make model
```

Install required Python dependencies:
```
pip install -r requirements.txt
```

Test out the models using the provided scripts:
```
python scripts/test_dynamics_model.py
python scripts/test_rollout_model.py
```
These scripts print information about the execution time for the compiled model
and the equivalent JAX model, and assert that both models produce equivalent
results.