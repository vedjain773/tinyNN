#ifndef INIT
#define INIT

#include <Eigen/Core>
#include <random>
#include <cmath>

using Eigen::Ref;
using Eigen::MatrixXd;

void Xavier (Ref<MatrixXd> weights, int fanIn, int fanOut);
void HeNormal(Ref<MatrixXd> weights, int fanIn);

#endif
