#ifndef LOSS
#define LOSS

#include <iostream>
#include <Eigen/Core>
using Eigen::MatrixXd;
using Eigen::Ref;

enum LossType {
    MSE,
    SCE
};

class Loss {
    public:
    virtual double calcGradient(const Ref<const MatrixXd> output, const Ref<const MatrixXd> desOutput, Ref<MatrixXd> gradient);
};

class MsE : public Loss {
    public:
    double calcGradient(const Ref<const MatrixXd> output, const Ref<const MatrixXd> desOutput, Ref<MatrixXd> gradient);
};

class SoftCE : public Loss {
    public:
    double calcGradient(const Ref<const MatrixXd> output, const Ref<const MatrixXd> desOutput, Ref<MatrixXd> gradient);
};

int networkGuess(const Ref<const MatrixXd> output);

#endif
