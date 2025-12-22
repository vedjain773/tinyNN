#include <Eigen/Core>
#include <network.hpp>

using Eigen::MatrixXd;

Network::Network(std::vector<int> nPL, std::vector<ActType> aTypes) {
    if (nPL.size() != aTypes.size()) {
        throw std::runtime_error(
            "Invalid network configuration: each layer must have a corresponding activation type "
            "(layers = " + std::to_string(nPL.size()) +
            ", activation types = " + std::to_string(aTypes.size()) + ")"
        );
    }

    neuronsPerLayer = nPL;

    for (int i = 1; i < nPL.size(); i++) {
        std::cout << neuronsPerLayer.at(i) << " " << neuronsPerLayer.at(i-1) << std::endl;

        switch(aTypes.at(i)) {
            case SIGMOID:
            {
                Layer layer(neuronsPerLayer.at(i), neuronsPerLayer.at(i-1), &sig);
                layers.push_back(layer);
            }
            break;

            case RELU:
            {
                Layer layer(neuronsPerLayer.at(i), neuronsPerLayer.at(i-1), &relu);
                layers.push_back(layer);
            }
            break;

            case NONE:
            {
                Layer layer(neuronsPerLayer.at(i), neuronsPerLayer.at(i-1), &none);
                layers.push_back(layer);
            }
            break;
        }

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

void Network::save(const std::string path) {
    std::ofstream out(path, std::ios::binary);

    const char head[4] = {'N', 'N', 'E', 'T'};
    int version = 1;
    int layerCount = neuronsPerLayer.size();

    out.write(head, 4);
    out.write(reinterpret_cast<char*>(&version), sizeof(int));
    out.write(reinterpret_cast<char*>(&layerCount), sizeof(int));

    int inpNeurons = inputLayer.size;

    out.write(reinterpret_cast<char*>(&inpNeurons), sizeof(int));

    for (const Layer& layer : layers) {
        int noOfNeurons = layer.size;

        out.write(reinterpret_cast<char*>(&noOfNeurons), sizeof(int));

        out.write(reinterpret_cast<const char*>(layer.weights.data()), sizeof(double) * layer.size * layer.prevLayerSize);
        out.write(reinterpret_cast<const char*>(layer.biases.data()), sizeof(double) * layer.size);
    }
}

void Network::load(const std::string path) {
    std::ifstream in(path, std::ios::binary);

    char head[4];
    int version;
    int layerCount;

    in.read(head, 4);
    in.read(reinterpret_cast<char*>(&version), sizeof(int));
    in.read(reinterpret_cast<char*>(&layerCount), sizeof(int));

    if (layerCount != neuronsPerLayer.size()) {
        throw std::runtime_error(
            "Network architecture mismatch: expected " + std::to_string(layerCount) +
            " layers, but network specifies " +
            std::to_string(neuronsPerLayer.size())
        );
    }

    int inpNeurons = 0;

    in.read(reinterpret_cast<char*>(&inpNeurons), sizeof(int));

    if (inpNeurons != inputLayer.size) {
        throw std::runtime_error(
            "Layer size mismatch: expected " + std::to_string(inpNeurons) +
            " neurons, but layer contains " + std::to_string(inputLayer.size)
        );
    }

    for (Layer& layer : layers) {
        int noOfNeurons;

        in.read(reinterpret_cast<char*>(&noOfNeurons), sizeof(int));

        if (noOfNeurons != layer.size) {
            throw std::runtime_error(
                "Layer size mismatch: expected " + std::to_string(noOfNeurons) +
                " neurons, but layer contains " + std::to_string(layer.size)
            );
        }

        in.read(reinterpret_cast<char*>(layer.weights.data()), sizeof(double) * layer.size * layer.prevLayerSize);
        in.read(reinterpret_cast<char*>(layer.biases.data()), sizeof(double) * layer.size);
    }
}
