#include <Eigen/Core>
#include <trainer.hpp>
#include <network.hpp>
#include <loss.hpp>
#include <readCSV.hpp>

Trainer::Trainer(int sSize, int eSize, int bSize, double lRate) {
    sampleSize = sSize;
    epochSize = eSize;
    batchSize = bSize;
    learningRate = lRate;
}

void Trainer::recordLossPerBatch(double loss, int epochNum, int batchNum, std::string path) {
    std::ofstream out(path, std::ios::app);

    out << epochNum * (sampleSize / batchSize) + batchNum << "," << loss << std::endl;
}

void Trainer::recordAccPerBatch(double acc, int epochNum, int batchNum, std::string path) {
    std::ofstream out(path, std::ios::app);

    out << epochNum * (sampleSize / batchSize) + batchNum << "," << acc << std::endl;
}

std::vector<std::vector <float>> Trainer::loadDataSet(std::string path, int noOfSamples) {
    std::vector<std::vector <float>> samples;
    samples.reserve(sampleSize);

    std::cout << "Loading data..." << std::endl;
    samples = readFile(path, noOfSamples);
    std::cout << "Loaded data" << std::endl;

    return samples;
}

std::vector<int> Trainer::shuffle() {
    std::vector<int> indices;

    for (int i = 0; i < sampleSize; i++) {
        indices.push_back(i);
    }

    std::random_shuffle(indices.begin(), indices.end());

    return indices;
}

void Trainer::trainModel(const std::string path, Network& network) {

    std::vector<std::vector <float>> samples = loadDataSet(path, 60000);
    std::cout << "Sample Size: " << sampleSize << "\n";
    std::cout << "Epoch Size: " << epochSize << "\n";
    std::cout << "Batch Size: " << batchSize << "\n";
    std::cout << "Learning Rate: " << learningRate << "\n";

    Eigen::MatrixXd op = Eigen::MatrixXd::Zero(10, 1);
    Eigen::MatrixXd desOp = Eigen::MatrixXd::Zero(10, 1);
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(10, 1);

    const double avgMultiplier = 0.01;
    double loss;

    for (int epoch = 0; epoch < epochSize; epoch++) {
        std::cout << "      Epoch Number:       " << epoch << std::endl;

        std::vector<int> shuffleIndices = shuffle();

        for (int i = 0; i < sampleSize; i += batchSize) {
            int acc = 0;
            double loss = 0;

            for (int j = i; j < i + batchSize; j++) {
                desOp = Eigen::MatrixXd::Constant(10, 1, 0);
                std::vector<float> data = samples.at(shuffleIndices.at(j));

                double label = 0.0;
                std::vector<float> initialVals = sliceVector(data, label);

                network.fPass(initialVals, op);

                int netGuess = networkGuess(op);

                desOp((int)label, 0) = 1;

                loss += softCEGrad(op, desOp, grad);
                network.bPass(grad);

                if (netGuess == (int)label) {
                    acc++;
                }
            }

            for (int i = network.layers.size()-1; i >= 0; i--) {
                network.layers.at(i).updateParams(learningRate, avgMultiplier);
            }

            for (int i = network.layers.size()-1; i >= 0; i--) {
                network.layers.at(i).resetGrads();
            }

            int batchNum = (int)(i / batchSize);

            if (batchNum % 60 == 0) {
                std::cout << "Batch Number: " << batchNum << " Accuracy: " << acc << "/" << batchSize << " Loss: " << 0.01 * loss << std::endl;
            }

            recordAccPerBatch(0.01 * acc, epoch, batchNum, "./logs/log1/acc.csv");
            recordLossPerBatch(0.01 * loss, epoch, batchNum, "./logs/log1/loss.csv");
        }
    }
}

void Trainer::testModel(const std::string path, Network& network) {
    std::vector<std::vector <float>> samples = loadDataSet(path, 10000);
    std::cout << "Learning Rate: " << learningRate << "\n";

    Eigen::MatrixXd op = Eigen::MatrixXd::Zero(10, 1);
    Eigen::MatrixXd desOp = Eigen::MatrixXd::Zero(10, 1);

    // std::vector<int> shuffleIndices = shuffle();

    int acc = 0;
    for (int i = 0; i < 10000; i += 1) {

        desOp = Eigen::MatrixXd::Zero(10, 1);
        std::vector<float> data = samples.at(i);

        double label = 0.0;
        std::vector<float> initialVals = sliceVector(data, label);

        network.fPass(initialVals, op);

        int netGuess = networkGuess(op);
        desOp((int)label, 0) = 1.0;

        if (netGuess == (int)label) {
            acc++;
        }

    }

    std::cout << "Accuracy: " << acc << "/" << 10000 << std::endl;
}
