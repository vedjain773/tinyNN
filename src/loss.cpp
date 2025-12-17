#include <loss.hpp>
#include <Eigen/Core>
using Eigen::MatrixXd;
using Eigen::Ref;

double calcGradient(const Ref<const MatrixXd> output, const Ref<const MatrixXd> desOutput, Ref<MatrixXd> gradient) {
    int rows = output.rows();
    int cols = output.cols();
    double loss = 0;

    gradient = 2 * (output - desOutput);

    for (int i = 0; i < rows; i++) {
        loss += gradient(i, 0) * gradient(i, 0);
    }

    return loss / 4;
}

int networkGuess(const Ref<const MatrixXd> output) {
    double max_term = 0.0;
    int max_index = 0;

    for (int i = 0; i < output.rows(); i++) {
        if (max_term < output(i, 0)) {
            max_term = output(i, 0);
            max_index = i;
        }
    }

    return max_index;
}
