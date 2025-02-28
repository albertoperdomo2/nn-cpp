# C++ Neural Network implementation from scratch

This project is a lightweight neural network mini-library built from scratch in C++ for educational purposes. The goal is to gain better understanding of the fundamental components of neural networks without relying on existing libraries or frameworks.

## Components

- `Matrix`: A templated class for matrix operations
- `Activation`: Various activation functions (ReLU, Sigmoid, Tanh, LeakyReLU)
- `Layer`: Neural network layer with forward/backward propagation
- `Optimizer`: Gradient descent optimization (SGD with momentum)
- `Network`: Management of multiple layers for training

## Features

- Weight initialization strategies (Xavier/Glorot, He)
- Configurable network architecture
- Backpropagation implementation
- Support for basic classification and regression tasks
- XOR problem training test

## Learning Focus

This implementation prioritizes clarity and understanding over performance. The matrix operations are intentionally simple and straightforward to make the neural network concepts more digestible and do not take advantage of hardware (e.g. no SIMD).

## Performance

This mini-library is - evidently - not optimized for performance. Limitations include (among a very long list):

- Basic matrix operations without hardware acceleration **!!!**
- No parallel processing or GPU support
- Minimal memory optimization
- Limited to dense, fully-connected layers
- Not designed for large datasets or deep architectures

## Next steps

I consider adding a bunch of things to make it a more complete learning resource:
- Additional optimizer algorithms (Adam, RMSprop)
- More layer types (Convolutional, Pooling)
- Batch normalization
- Performance optimizations
