#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>

#include "rigid_body.h"
#include "types.h"

namespace ad = CppADCodeGenEigenPy;

// * recall that this is a **cost**
// * input is the array of inputs
// * paramters are mass, inertia, x0, xd's

const double TIMESTEP = 0.1;
const size_t NUM_TIME_STEPS = 10;
const size_t NUM_INPUT = INPUT_DIM * NUM_TIME_STEPS;

template <typename Scalar>
struct RolloutCostModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADScalar;
    using typename ad::ADModel<Scalar>::ADVector;
    using typename ad::ADModel<Scalar>::ADMatrix;

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

    // Get initial state from parameter vector
    ADVector initial_state(const ADVector& parameters) const {
        return parameters.head(STATE_DIM);
    }

    ADVector desired_state(const ADVector& parameters) const {
        return parameters.tail(NUM_TIME_STEPS * STATE_DIM);
    }

    Scalar get_mass(const ADVector& parameters) const { return parameters(0); }

    Mat3<ADScalar> get_inertia(const ADVector& parameters) const {
        ADVector inertia_vec = parameters.segment(1, 3 * 3);
        Eigen::Map<Mat3<ADScalar>> inertia(inertia_vec.data(), 3, 3);
        return inertia;
    }

    std::vector<StateVec<ADScalar>> get_desired_states(
        const ADVector& parameters) {}

    StateVec<ADScalar> get_initial_state(const ADVector& input) {
        return input.head(STATE_DIM);
    }

    std::vector<InputVec<ADScalar>> get_wrenches(const ADVector& input) {
        std::vector<InputVec<ADScalar>> us;
        // TODO
        return us;
    }

    /**
     * Compute cost of the rollout.
     */
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        ADScalar mass = get_mass(parameters);
        Mat3<ADScalar> inertia = get_inertia(parameters);

        // TODO recall there were problems with using fixed-sized matrices with
        // AD
        std::vector<StateVec<ADScalar>> xds = get_desired_states(parameters);
        StateVec<ADScalar> x0 = get_initial_state(input);
        std::vector<InputVec<ADScalar>> us = get_wrenches(input);

        // Do the rollout
        RigidBody<ADScalar> body(mass, inertia);
        std::vector<StateVec<ADScalar>> xs =
            body.rollout(x0, us, ADScalar(TIMESTEP), NUM_TIME_STEPS);

        // Compute cost
        // TODO need to compute state error vector
        ADVector cost = ADVector::Zero(1);

        for (int i = 0; i < NUM_TIME_STEPS; ++i) {
            x_err = ...;
            cost_i = 0.5 * (x_err.transpose() * x_err + 0.1 * u.transpose() * u);

            cost(0) += cost_i
        }
        return cost;
    }
};
