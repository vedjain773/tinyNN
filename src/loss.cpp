#include <loss.hpp>
#include <Eigen/Core>
using Eigen::MatrixXd;
using Eigen::Ref;

double Loss::calcGradient(const Ref<const MatrixXd> output, const Ref<const MatrixXd> desOutput, Ref<MatrixXd> gradient) {
    int rows = output.rows();
    double loss = 0;

    gradient = 2 * (output - desOutput);

    for (int i = 0; i < rows; i++) {
        loss += gradient(i, 0) * gradient(i, 0);
    }

    return loss / 4;
}

double MsE::calcGradient(const Ref<const MatrixXd> output, const Ref<const MatrixXd> desOutput, Ref<MatrixXd> gradient) {
    int rows = output.rows();
    double loss = 0;

    gradient = 2 * (output - desOutput);

    for (int i = 0; i < rows; i++) {
        loss += gradient(i, 0) * gradient(i, 0);
    }

    return loss / 4;
}

double SoftCE::calcGradient(const Ref<const MatrixXd> output, const Ref<const MatrixXd> desOutput, Ref<MatrixXd> gradient) {
    int rows = output.rows();
    MatrixXd softMax = output;

    double maxLogit = output.maxCoeff();

    double sum = 0.0;
    for (int i = 0; i < rows; ++i) {
        sum += std::exp(output(i, 0) - maxLogit);
    }

    for (int i = 0; i < rows; ++i) {
        softMax(i, 0) = std::exp(output(i, 0) - maxLogit) / sum;
    }

    gradient = (softMax - desOutput);

    double loss = 0.0;

    for (int i = 0; i < rows; ++i) {
        if (desOutput(i, 0) == 1.0) {
            loss = -std::log(softMax(i, 0));
            break;
        }
    }

    return loss;
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
