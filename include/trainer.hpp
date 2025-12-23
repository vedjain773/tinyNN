#ifndef TRAINER
#define TRAINER

#include <network.hpp>
#include <loss.hpp>
#include <readCSV.hpp>
#include <vector>
#include <stdexcept>

class Trainer {
    private:
    MsE mse;
    SoftCE sce;

    public:
    int sampleSize;
    int epochSize;
    int batchSize;
    double learningRate;

    Loss* lossNet;

    Trainer(int sampleSize, int epochSize, int batchSize, double learningRate, LossType lt);

    std::vector<std::vector <float>> loadDataSet(std::string path, int noOfSamples);
    std::vector<int> shuffle();
    void trainModel(const std::string path, Network& network);
    void testModel(const std::string path, Network& network);

    void recordLossPerBatch(double loss, int epochNum, int batchNum, std::string path);
    void recordAccPerBatch(double acc, int epochNum, int batchNum, std::string path);
};

#endif
