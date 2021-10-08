#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>

namespace ad = CppADCodeGenEigenPy;

const size_t STATE_DIM = 7 + 6;
const size_t INPUT_DIM = 6;
const size_t NUM_TIME_STEPS = 1;  // 10
const size_t NUM_INPUT = INPUT_DIM * NUM_TIME_STEPS;

template <typename Scalar>
struct ForwardRolloutModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADScalar;
    using typename ad::ADModel<Scalar>::ADVector;
    using typename ad::ADModel<Scalar>::ADMatrix;

    using Mat3 = Eigen::Matrix<ADScalar, 3, 3>;

    // Generate the input to the function
    // In this example, the input is the force and torque at each timestep
    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    ADVector parameters() const override {
        // Parameter is the initial state
        ADVector x0(STATE_DIM);
        // clang-format: off
        x0 << ADVector::Ones(3),                 // position
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

    ADVector position(const ADVector& x) const {
        return x.head(3);
    }

    ADVector angular_velocity(const ADVector& x) const {
        return x.segment(7, 3);
    }

    // Evaluate the function
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        // Dynamic parameters of the body
        ADScalar mass(1.0);
        Mat3 inertia = Mat3::Identity();
        Mat3 I_inv = inertia.inverse();

        ADVector x0 = initial_state(parameters);
        ADVector xd = desired_state(parameters);


        Vec3 force = input.head(3);
        Vec3 torque = input.tail(3);
        Vec3 omega = angular_velocity(x0);

        // Newton-Euler equations
        Vec3 acc_linear = force / mass;
        Vec3 acc_angular = I_inv * (torque - omega.cross(inertia * omega));

        // Integrate


        ADVector cost(1);
        cost << ADScalar(0);
        return cost;
    }

    // ADScalar mass;
    // ADMatrix inertia;
};