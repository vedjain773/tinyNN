#include <activation.hpp>
#include <math.h>

MatrixXd Activation::activate(const Ref<const MatrixXd> preAct) {
    return preAct;
}

MatrixXd Activation::derivative(const Ref<const MatrixXd> inp) {
    return inp;
}

void Activation::init(Ref<MatrixXd> weights, int fanIn, int fanOut){
    //
}

MatrixXd Sigmoid::activate(const Ref<const MatrixXd> preAct) {
    MatrixXd out = preAct;

    for (int i = 0; i < preAct.size(); i++) {
        double num = exp(preAct(i, 0));
        double denom = 1 + num;

        double sigmoid = num / denom;
        out(i, 0) = sigmoid;
    }

    return out;
}

MatrixXd Sigmoid::derivative(const Ref<const MatrixXd> inp) {
    MatrixXd out = inp;

    for (int i = 0; i < inp.size(); i++) {
        double num = exp(inp(i, 0));
        double denom = 1 + num;

        double sigmoid = num / denom;
        out(i, 0) = sigmoid * (1 - sigmoid);
    }

    return out;
}

void Sigmoid::init(Ref<MatrixXd> weights, int fanIn, int fanOut){
    Xavier(weights, fanIn, fanOut);
}

MatrixXd Relu::activate(const Ref<const MatrixXd> preAct) {
    MatrixXd out = preAct;

    for (int i = 0; i < preAct.size(); i++) {
        out(i, 0) = preAct(i, 0) > 0 ? preAct(i, 0) : 0.0;
    }

    return out;
}

MatrixXd Relu::derivative(const Ref<const MatrixXd> inp) {
    MatrixXd out = inp;

    for (int i = 0; i < inp.size(); i++) {
        out(i, 0) = inp(i, 0) > 0 ? 1.0 : 0.0;
    }

    return out;
}

void Relu::init(Ref<MatrixXd> weights, int fanIn, int fanOut){
    HeNormal(weights, fanIn);
}
