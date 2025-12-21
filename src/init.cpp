#include <init.hpp>
#include <Eigen/Core>
#include <cmath>
#include <random>

using Eigen::Ref;
using Eigen::MatrixXd;

void Xavier (Ref<MatrixXd> weights, int fanIn, int fanOut) {
    double stddev = std::sqrt(2.0 / (fanIn + fanOut));

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, stddev);

    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            weights(i, j) = distribution(generator);
        }
    }
}

void HeNormal(Ref<MatrixXd> weights, int fanIn) {
    double stddev = std::sqrt(2.0 / fanIn);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, stddev);

    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            weights(i, j) = distribution(generator);
        }
    }
}
