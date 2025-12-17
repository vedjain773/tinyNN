#include <Eigen/Core>
#include <iostream>
#include <trainer.hpp>

int main() {
    // std::cout << "Hello Worlds!" << std::endl;

    std::vector<int> arch = {784, 32, 10};
    Network network(arch);

    Trainer trainer(60000, 1, 100, 0.035);
    trainer.trainModel("data/mnist_train.csv/mnist_train.csv", network);

    std::cout << "Starting Test phase... \n";

    trainer.testModel("./data/mnist_test.csv/mnist_test.csv", network);

    return 0;
}
