#ifndef LOSS
#define LOSS

#include <iostream>
#include <Eigen/Core>
using Eigen::MatrixXd;
using Eigen::Ref;

double calcGradient(const Ref<const MatrixXd> output, const Ref<const MatrixXd> desOutput, Ref<MatrixXd> gradient);
int networkGuess(const Ref<const MatrixXd> output);

#endif
