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


## MNIST example

With this simple mini-library, I was able to write a simple handwritten digits classifier using the MNIST dataset. 
In order to train the network on the MNIST dataset, first the dataset has to be downloaded with `make setup`. Then, to build the entire project, you can run `make build`. Once the build is complete, you can run the command from the Make logs e.g. `./build/mnist`. It should be executed from the repository's root since it has a hardcoded path for the data). 

When the training starts, you should see training metrics such as:

```
Loading MNIST dataset...
Loaded 1000 training images and 100 test images.

Training network...

Epoch 1, Batch 0, Loss: 0.134804
Epoch 1, Batch 10, Loss: 0.0772555
Epoch 1, Batch 20, Loss: 0.0626018
Epoch 1, Batch 30, Loss: 0.0537534
Epoch 1/10, Loss: 0.0536376, Accuracy: 62%
...
```

and when the evaluation is running, there is even some ASCII art to visualize the digits :)

```
-------------------------
Actual: 4, Predicted: 4
Confidence:
0: 0.0005% 
1: 0.0158% 
2: 0.0000%
3: 0.2409%
4: 99.8367%
5: 0.3980%
6: 0.0014%
7: 0.0001%
8: 0.0085% 
9: 0.1137%
Image:                                                                                                                                                                                      
          *+.         - 
         ##+         *#. 
        *#+         +#- 
       .##         .#+ 
       +#-        .##-
      +#*         -#+ 
      +#+        +## 
      +##*+....+*##+ 
      .*##########*. 
        .+##+++.##+ 
               .#* 
               +#. 
              -##. 
              +#+ 
              ##. 
             -## 
             *#+  .   
             *#+-*+ 
             *###+.         
             .*+            
                             
-------------------------
```

## Tests

Tests are build with the main build, but in case you only want to build the tests, you can run `make tests` and then, you should be able to run `./tests/build/tests/network_tests`, for example.


## Next steps

I consider adding a bunch of things to make it a more complete learning resource:
- Additional optimizer algorithms (Adam, RMSprop)
- More layer types (Convolutional, Pooling)
- Batch normalization
- Performance optimizations
