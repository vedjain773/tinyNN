#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <trainer.hpp>

int main() {
    std::vector<int> arch = {784, 48, 10};
    Network network(arch);

    //Training the model
    Trainer trainer(60000, 40, 100, 0.0175);
    trainer.trainModel("data/mnist_train.csv/mnist_train.csv", network);

    //Saving the model
    std::cout << "Saving model... \n";
    network.save("./saves/mnist1.bin");

    std::vector<int> arch = {784, 48, 10};
    Network network2(arch);

    //Loading the saved model onto another model
    std::cout << "Loading model... \n";
    network2.load("./saves/mnist1.bin");

    //Testing the loaded model
    std::cout << "Starting Test phase... \n";
    trainer.testModel("./data/mnist_test.csv/mnist_test.csv", network2);

    return 0;
}
