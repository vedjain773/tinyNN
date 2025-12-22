#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include <iostream>
#include <init.hpp>
#include <Eigen/Core>

using Eigen::MatrixXd;
using Eigen::Ref;

enum ActType {
    SIGMOID,
    RELU,
    NONE
};

class Activation {
    public:
    virtual MatrixXd activate(const Ref<const MatrixXd> preAct);
    virtual MatrixXd derivative(const Ref<const MatrixXd> inp);
    virtual void init (Ref<MatrixXd> weights, int fanIn, int fanOut);
};

class Sigmoid: public Activation {
    public:
    MatrixXd activate(const Ref<const MatrixXd> preAct);
    MatrixXd derivative(const Ref<const MatrixXd> inp);
    void init (Ref<MatrixXd> weights, int fanIn, int fanOut);
};

class Relu: public Activation {
    public:
    MatrixXd activate(const Ref<const MatrixXd> preAct);
    MatrixXd derivative(const Ref<const MatrixXd> inp);
    void init (Ref<MatrixXd> weights, int fanIn, int fanOut);
};

class None: public Activation {
    public:
    MatrixXd activate(const Ref<const MatrixXd> preAct);
    MatrixXd derivative(const Ref<const MatrixXd> inp);
    void init (Ref<MatrixXd> weights, int fanIn, int fanOut);
};

#endif
