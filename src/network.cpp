#include <Eigen/Core>
#include <network.hpp>

using Eigen::MatrixXd;

Network::Network(std::vector<int> nPL) {
    neuronsPerLayer = nPL;

    for (int i = 1; i < nPL.size(); i++) {
        std::cout << neuronsPerLayer.at(i) << " " << neuronsPerLayer.at(i-1) << std::endl;
        Layer layer(neuronsPerLayer.at(i), neuronsPerLayer.at(i-1));
        layers.push_back(layer);
    }

    inputLayer.resize(neuronsPerLayer.at(0));
}

void Network::fPass(std::vector<float> inpLayerVals, Ref<MatrixXd> op) {
    inputLayer.initInputLayer(inpLayerVals);
    MatrixXd prevLayerActs = inputLayer.inpAct;

    for (int i = 0; i < layers.size(); i++) {
        layers[i].forwardPass(prevLayerActs);
        prevLayerActs = layers[i].activations;
    }

    op = layers.at(layers.size() - 1).activations;
    // std::cout << op << "\n\n";
}

void Network::bPass(const Ref<const MatrixXd> grad) {
    MatrixXd gradient = grad;

    for (int i = layers.size()-1; i >= 1; i--) {
        layers.at(i).backwardPass(gradient, layers.at(i-1).activations);
        gradient = layers.at(i).gAct;
    }
}
