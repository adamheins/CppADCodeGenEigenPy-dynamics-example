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
If you get an error about not being able to find an Eigen header, you may have
to change the path to the Eigen include directory in the Makefile.

Install required Python dependencies (note that Python 3 is expected):
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

On my system, loading the C++ `RolloutCostModel` takes under 1 millisecond,
whereas the equivalent JAX model takes about 5 seconds, since it has to JIT
compile each time the script is run. After the initial compilation, evaluating
the Jacobians is also about an order of magnitude faster using the C++ model.
