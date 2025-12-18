#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Core>
#include <init.hpp>
#include <activation.hpp>
#include <vector>
using Eigen::MatrixXd;
using Eigen::Ref;

class Layer {
    public:
    int size;
    int prevLayerSize;

    MatrixXd preActivations;
    MatrixXd activations;
    MatrixXd weights;
    MatrixXd biases;

    MatrixXd gAct;
    MatrixXd gWeights;
    MatrixXd gBiases;

    Layer (int noOfNeurons, int prevLayerNeurons);

    void forwardPass(const Ref<const MatrixXd>& prevLayerAct);
    void backwardPass(const Ref<const MatrixXd> gradient, const Ref<const MatrixXd> prevLayerAct);
    void updateParams(double learningRate, double multiplier);
    void resetGrads();
};

class InputLayer {
    public:
    int size;

    MatrixXd inpAct;

    InputLayer ();
    InputLayer (int rows);

    void resize(int rows);
    void initInputLayer (std::vector<float> pixelVals);
};

#endif
