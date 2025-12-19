# Tiny NN
This project is a from-scratch implementation of a fully connected neural network written in C++, using Eigen for linear algebra with a great emphasis on user configuration and customisation.

## Network Configuration
The project allows explicit and flexible configuration of neural network architectures directly in code.

### Supported configuration options include:
- Arbitrary number of layers
- Configurable number of neurons per layer
- Explicit layer ordering
- Adjustable training hyperparameters (learning rate, batch size, epochs)

Network architectures are defined programmatically by constructing and assembling layers, making it straightforward to modify or extend existing configurations.

## Model Saving and Loading
The project supports serialization and deserialization of trained neural networks.

### Saved model data includes:
- Number of layers
- Layer dimensions
- Layer types
- Weight matrices
- Bias vectors

Model loading requires a pre-defined architecture that matches the saved configuration.
