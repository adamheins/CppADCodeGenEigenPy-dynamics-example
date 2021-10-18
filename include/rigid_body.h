#pragma once

#include <Eigen/Eigen>

#include "types.h"

template <typename Scalar>
class RigidBody {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RigidBody(const Scalar mass, const Mat3<Scalar>& inertia)
        : mass(mass), inertia(inertia), inertia_inv(inertia.inverse()) {}

    // Compute forward dynamics for the rigid body. Force inputs are assumed to
    // be applied in the inertial (world) frame.
    Vec6<Scalar> forward_dynamics(const StateVec<Scalar>& x,
                                  const InputVec<Scalar>& u) const {
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

    // Roll out the forward dynamics over num_steps time steps, each spaced dt
    // seconds apart.
    std::vector<StateVec<Scalar>> rollout(const StateVec<Scalar>& x0,
                 const std::vector<InputVec<Scalar>>& us, const Scalar dt,
                 const size_t num_steps) const {
        StateVec<Scalar> x = x0;
        std::vector<StateVec<Scalar>> xs;
        for (InputVec<Scalar> u : us) {
            Vec6<Scalar> A = forward_dynamics(x, u);

            // Integrate linear portion of state
            Vec3<Scalar> r0 = position(x);
            Vec3<Scalar> v0 = linear_velocity(x);
            Vec3<Scalar> a = A.head(3);

            Vec3<Scalar> r1 = r0 + dt * v0 + 0.5 * dt * dt * a;
            Vec3<Scalar> v1 = v0 + dt * a;

            // Integrate rotation portion
            Eigen::Quaternion<Scalar> q0 = orientation(x);
            Vec3<Scalar> omega0 = angular_velocity(x);
            Vec3<Scalar> alpha = A.tail(3);
            Vec3<Scalar> omega1 = omega0 + dt * alpha;

            // First term of the Magnus expansion
            Vec3<Scalar> aa_vec = 0.5 * dt * (omega0 + omega1);

            // Map to a quaternion via exponential map
            Scalar angle = aa_vec.norm();
            Vec3<Scalar> axis = aa_vec.normalized();
            Eigen::AngleAxis<Scalar> aa(angle, axis);
            Eigen::Quaternion<Scalar> qw(aa);
            Eigen::Quaternion<Scalar> q1 = qw * q0;

            x << r1, q1, v1, omega1;

            xs.push_back(x);
        }
        return xs;
    }

   private:
    Vec3<Scalar> position(const StateVec<Scalar>& x) const {
        return x.head(3);
    }

    Vec3<Scalar> orientation(const StateVec<Scalar>& x) const {
        return x.segment(3, 4);
    }

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
