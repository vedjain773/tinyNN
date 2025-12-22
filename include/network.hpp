#ifndef NETWORK
#define NETWORK

#include "activation.hpp"
#include <Eigen/Core>
#include <fstream>
#include <loss.hpp>
#include <layer.hpp>
#include <vector>

using Eigen::MatrixXd;

class Network {
    private:
    Sigmoid sig;
    Relu relu;
    None none;

    public:
    std::vector<int> neuronsPerLayer;
    std::vector<Layer> layers;
    InputLayer inputLayer;

    Network(std::vector<int> nPL, std::vector<ActType> aTypes);

    void fPass(std::vector<float> inpLayerVals, Ref<MatrixXd> op);
    void bPass(const Ref<const MatrixXd> grad);

    void save(const std::string path);
    void load(const std::string path);
};

#endif
