#include <CppADCodeGenEigenPy/ADModel.h>
// #include <boost/filesystem.hpp>

#include "model.h"

namespace ad = CppADCodeGenEigenPy;

using Scalar = double;

static const std::string MODEL_NAME = "ForwardDynamicsModel";
static const std::string DIRECTORY_PATH = "/tmp/CppADCodeGenEigenPy";

static const Scalar MASS = 1.0;
static const Mat3<Scalar> INERTIA = Mat3<Scalar>::Identity();

int main() {
    // TODO we can pass in MODEL_NAME and DIRECTORY_PATH as arguments, and
    // assume the directory exists
    // boost::filesystem::create_directories(DIRECTORY_PATH);
    ForwardDynamicsModel<Scalar>(MASS, INERTIA)
        .compile(MODEL_NAME, DIRECTORY_PATH, ad::DerivativeOrder::Second);
}
