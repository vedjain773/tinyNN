#include <Eigen/Core>
#include <layer.hpp>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::Ref;

Layer::Layer(int noOfNeurons, int prevLayerNeurons) {
    size = noOfNeurons;
    prevLayerSize = prevLayerNeurons;

    preActivations = MatrixXd::Zero(noOfNeurons, 1);
    activations = MatrixXd::Zero(noOfNeurons, 1);
    weights = MatrixXd::Zero(noOfNeurons, prevLayerNeurons);
    biases = MatrixXd::Zero(noOfNeurons, 1);

    Xavier(weights, prevLayerNeurons, noOfNeurons);

    gAct = MatrixXd::Zero(noOfNeurons, 1);
    gWeights = MatrixXd::Zero(noOfNeurons, prevLayerNeurons);
    gBiases = MatrixXd::Zero(noOfNeurons, 1);
}

void Layer::forwardPass(const Ref<const MatrixXd>& prevLayerAct) {
    if (prevLayerAct.rows() != weights.cols()) {
        std::cout << "Invalid matrix product " << prevLayerAct.rows() << " " << weights.cols() << std::endl;
    } else {
        preActivations = (weights * prevLayerAct) + biases;

        for (int i = 0; i < activations.rows(); i++) {
            activations(i, 0) = sigmoid(preActivations(i, 0));
        }
    }
}

void Layer::backwardPass(const Ref<const MatrixXd> gradient, const Ref<const MatrixXd> prevLayerAct) {
    MatrixXd sigDZ;
    sigDZ = MatrixXd::Zero(gradient.rows(), 1);

    for (int i = 0; i < sigDZ.rows(); i++) {
        sigDZ(i, 0) = d_dtSigmoid(preActivations(i, 0));
    }

    MatrixXd commTerm = gradient.cwiseProduct(sigDZ);

    gWeights = gWeights + commTerm * prevLayerAct.transpose();
    gBiases = gBiases + commTerm ;
    gAct = weights.transpose() * commTerm;
}

void Layer::updateParams(double learningRate, double multiplier) {
    gWeights = multiplier * gWeights;
    gBiases = multiplier * gBiases;

    weights = weights - learningRate * gWeights;
    biases = biases - learningRate * gBiases;
}

void Layer::resetGrads() {
    gWeights = MatrixXd::Zero(gWeights.rows(), gWeights.cols());
    gBiases = MatrixXd::Zero(gBiases.rows(), 1);
}

InputLayer::InputLayer() {
    inpAct = MatrixXd::Zero(784, 1);
    size = 784;
}

InputLayer::InputLayer(int rows) {
    inpAct = MatrixXd::Zero(rows, 1);
    size = rows;
}

void InputLayer::resize(int rows) {
    inpAct.resize(rows, 1);
    size = rows;
}

void InputLayer::initInputLayer(std::vector<float> pixelVals) {
    for (int i = 0; i < pixelVals.size(); i++) {
        inpAct(i, 0) = pixelVals.at(i) / 255;
    }
}
