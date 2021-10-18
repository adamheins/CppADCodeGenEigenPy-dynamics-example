#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>

namespace ad = CppADCodeGenEigenPy;

const size_t STATE_DIM = 7 + 6;
const size_t INPUT_DIM = 6;

template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using Vec6 = Eigen::Matrix<Scalar, 6, 1>;

template <typename Scalar>
using StateVec = Eigen::Matrix<Scalar, STATE_DIM, 1>;

template <typename Scalar>
using InputVec = Eigen::Matrix<Scalar, INPUT_DIM, 1>;

// Note that we need to specify row-major here to match the
// expected-convention for numpy, when passing in inertia parameters.
template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;

template <typename Scalar>
class RigidBody {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RigidBody(const Scalar mass, const Mat3<Scalar>& inertia)
        : mass(mass), inertia(inertia), inertia_inv(inertia.inverse()) {}

    // compute forward dynamics for the rigid body
    Vec6<Scalar> forward_dynamics(const StateVec<Scalar>& x,
                                  const InputVec<Scalar>& u) {
        Vec3<Scalar> force = u.head(3);
        Vec3<Scalar> torque = u.tail(3);
        Vec3<Scalar> v = linear_velocity(x);
        Vec3<Scalar> omega = angular_velocity(x);

        // Find acceleration from Newton-Euler equations
        Vec3<Scalar> a = force / mass;
        Vec3<Scalar> alpha =
            inertia_inv * (torque - omega.cross(inertia * omega));

        Vec6<Scalar> A;
        A << a, alpha;
        return A;
    }

   private:
    // Get linear velocity component of the state
    Vec3<Scalar> linear_velocity(const StateVec<Scalar>& x) const {
        return x.segment(7, 3);
    }

    // Get angular velocity component of the state
    Vec3<Scalar> angular_velocity(const StateVec<Scalar>& x) const {
        return x.segment(10, 3);
    }

    Scalar mass;
    Mat3<Scalar> inertia;
    Mat3<Scalar> inertia_inv;
};

template <typename Scalar>
struct DynamicsModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADScalar;
    using typename ad::ADModel<Scalar>::ADVector;
    using typename ad::ADModel<Scalar>::ADMatrix;

    // Generate the input used when differentiating the function
    ADVector input() const override {
        return ADVector::Ones(STATE_DIM + INPUT_DIM);
    }

    // Generate parameters used when differentiating the function
    ADVector parameters() const override {
        ADScalar mass(1.0);
        Mat3<ADScalar> inertia = Mat3<ADScalar>::Identity();

        ADVector p(1 + inertia.size());
        p << mass, Eigen::Map<ADVector>(inertia.data(), inertia.size());
        return p;
    }

    /**
     * Compute acceleration from current state and force input.
     */
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        // Parameters are mass and inertia matrix of the body
        ADScalar mass = parameters(0);
        ADVector inertia_vec = parameters.tail(3 * 3);
        Eigen::Map<Mat3<ADScalar>> inertia(inertia_vec.data(), 3, 3);

        // Input is (state, system input) = (x, u)
        ADVector x = input.head(STATE_DIM);
        ADVector u = input.tail(INPUT_DIM);

        RigidBody<ADScalar> body(mass, inertia);
        return body.forward_dynamics(x, u);
    }
};
