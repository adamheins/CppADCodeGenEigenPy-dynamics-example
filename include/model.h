#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>

namespace ad = CppADCodeGenEigenPy;

template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3>;

const size_t STATE_DIM = 7 + 6;
const size_t INPUT_DIM = 6;
const size_t NUM_TIME_STEPS = 1;  // 10
const size_t NUM_INPUT = INPUT_DIM * NUM_TIME_STEPS;

template <typename Scalar>
struct ForwardDynamicsModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADScalar;
    using typename ad::ADModel<Scalar>::ADVector;
    using typename ad::ADModel<Scalar>::ADMatrix;

    ForwardDynamicsModel(Scalar mass, const Mat3<Scalar>& inertia)
        : ad::ADModel<Scalar>(), mass(mass) {}

    // Generate the input to the function
    // In this example, the input is the force and torque at each timestep
    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    ADVector parameters() const override {
        // Parameter is the initial state
        ADVector x0(STATE_DIM);
        // clang-format: off
        x0 << ADVector::Ones(3),               // position
            ADVector::Zero(3), ADScalar(1.0),  // orientation (quaternion)
            ADVector::Ones(6);                 // twist
        // clang-format: on
        return x0;
    }

    // Evaluate the function
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        // TODO I want this to be a cost function
        ADVector x0 = parameters;

        ADVector cost(1);
        cost << ADScalar(0);
        return cost;
    }

    ADScalar mass;
    // ADMatrix inertia;
};
