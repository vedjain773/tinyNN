#ifndef TRAINER
#define TRAINER

#include <network.hpp>
#include <loss.hpp>
#include <readCSV.hpp>
#include <vector>
#include <stdexcept>

class Trainer {
    public:
    int sampleSize;
    int epochSize;
    int batchSize;
    double learningRate;

    Trainer(int sampleSize, int epochSize, int batchSize, double learningRate);

    std::vector<std::vector <float>> loadDataSet(std::string path, int noOfSamples);
    std::vector<int> shuffle();
    void trainModel(const std::string path, Network& network);
    void testModel(const std::string path, Network& network);

    void recordLossPerBatch(double loss, int epochNum, int batchNum, std::string path);
    void recordAccPerBatch(double acc, int epochNum, int batchNum, std::string path);
};

#endif
