#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>

namespace ad = CppADCodeGenEigenPy;

const size_t STATE_DIM = 7 + 6;
const size_t INPUT_DIM = 6;

template <typename Scalar>
struct DynamicsModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADScalar;
    using typename ad::ADModel<Scalar>::ADVector;
    using typename ad::ADModel<Scalar>::ADMatrix;

    using Vec3 = Eigen::Matrix<ADScalar, 3, 1>;
    using Mat3 = Eigen::Matrix<ADScalar, 3, 3, Eigen::RowMajor>;

    // Generate the input to the function
    // In this example, the input is the force and torque at each timestep
    ADVector input() const override {
        return ADVector::Ones(STATE_DIM + INPUT_DIM);
    }

    ADVector parameters() const override {
        ADScalar mass(1.0);
        Mat3 inertia = Mat3::Identity();

        ADVector p(1 + 3 * 3);
        p << mass, Eigen::Map<ADVector>(inertia.data(), inertia.size());
        return p;
    }

    // Get linear velocity component of the state
    ADVector linear_velocity(const ADVector& x) const {
        return x.segment(7, 3);
    }

    // Get angular velocity component of the state
    Vec3 angular_velocity(const ADVector& x) const {
        return x.segment(10, 3);
    }

    // Evaluate the function
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        // TODO maybe this just converts to accelerations? it should allow us
        // to compute df/dx and df du; that means we need to return f(x, u) =:
        // \dot{x}

        // Parameters are mass and inertia matrix of the body
        ADScalar mass = parameters(0);
        ADVector I_vec = parameters.tail(3 * 3);
        Eigen::Map<Mat3> inertia(I_vec.data(), 3, 3);
        Mat3 I_inv = inertia.inverse();

        // Input is (state, system input) = (x, u)
        ADVector x = input.head(STATE_DIM);
        ADVector u = input.tail(INPUT_DIM);

        Vec3 force = u.head(3);
        Vec3 torque = u.tail(3);
        Vec3 v = linear_velocity(x);
        Vec3 omega = angular_velocity(x);

        // Find acceleration from Newton-Euler equations
        Vec3 a = force / mass;
        Vec3 alpha = I_inv * (torque - omega.cross(inertia * omega));

        ADVector f(2 * INPUT_DIM);
        f << v, omega, a, alpha;
        return f;
    }
};
