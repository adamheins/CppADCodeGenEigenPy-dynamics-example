#include <CppADCodeGenEigenPy/ADModel.h>

#include "dynamics_model.h"
// #include "rollout_model.h"

namespace ad = CppADCodeGenEigenPy;

using Scalar = double;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: compile_model <directory>" << std::endl;
        return 1;
    }
    std::string directory_path = argv[1];

    DynamicsModel<Scalar> dynamics_model;
    // RolloutModel<Scalar> rollout_model(dynamics_model);

    dynamics_model.compile("DynamicsModel", directory_path,
                           ad::DerivativeOrder::Second);
    // rollout_model.compile("RolloutModel", directory_path,
    //                       ad::DerivativeOrder::Second);
}
