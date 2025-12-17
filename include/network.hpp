#ifndef NETWORK
#define NETWORK

#include <Eigen/Core>
#include <loss.hpp>
#include <layer.hpp>
#include <vector>

using Eigen::MatrixXd;

class Network {
    public:
    std::vector<int> neuronsPerLayer;
    std::vector<Layer> layers;
    InputLayer inputLayer;

    Network(std::vector<int> nPL);

    void fPass(std::vector<float> inpLayerVals, Ref<MatrixXd> op);
    void bPass(const Ref<const MatrixXd> grad);
};

#endif
