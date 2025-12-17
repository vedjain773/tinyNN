#include <activation.hpp>
#include <math.h>

double sigmoid(double x) {
    double num = exp(x);
    double denom = exp(x) + 1;

    if (denom != 0.0) {
        return (num / denom);
    } else  {
        return 0.0;
    }
}

double d_dtSigmoid(double x) {
    return (sigmoid(x) * (1 - sigmoid(x)));
}

double ReLU(double x) {
    return ((x > 0) ? x : 0);
}

double d_dtReLU(double x) {
    return ((x > 0) ? 1 : 0);
}
