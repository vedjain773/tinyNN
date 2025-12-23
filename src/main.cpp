#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <trainer.hpp>

int main() {
    std::vector<ActType> types = {SIGMOID, SIGMOID, SIGMOID, SIGMOID};
    std::vector<int> arch = {784, 128, 64, 10};
    Network network(arch, types);

    //Training the model
    Trainer trainer(60000, 10, 100, 0.01, MSE);
    trainer.trainModel("data/mnist_train.csv/mnist_train.csv", network);

    //Saving the model
    std::cout << "Saving model... \n";
    network.save("./saves/mnist4.bin");

    std::vector<ActType> types2 = {SIGMOID, SIGMOID, SIGMOID, SIGMOID};
    std::vector<int> arch2 = {784, 128, 64, 10};
    Network network2(arch2, types2);

    //Loading the saved model onto another model
    std::cout << "Loading model... \n";
    network2.load("./saves/mnist4.bin");

    //Testing the loaded model
    std::cout << "Starting Test phase... \n";
    trainer.testModel("./data/mnist_test.csv/mnist_test.csv", network2);

    return 0;
}
