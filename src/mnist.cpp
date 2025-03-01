#include <iostream>
#include <iomanip>
#include <cstring>
#include "nn/network.hpp"
#include "nn/layer.hpp"
#include "nn/optimizer.hpp"
#include "nn/activation.hpp"
#include "mnist_utils.cpp"

int main() {
    // Create a network for digit recognition
    nn::Network<float> network;
    
    // Create layers: 784 input neurons for 28x28 images, 10 output classes
    auto* layer1 = new nn::Layer<float, nn::activations::ReLU>(784, 128);
    auto* layer2 = new nn::Layer<float, nn::activations::ReLU>(128, 64);
    auto* layer3 = new nn::Layer<float, nn::activations::Sigmoid>(64, 10);
    
    // Add layers to the network
    network.add(layer1);
    network.add(layer2);
    network.add(layer3);
    
    auto* optimizer1 = new nn::SGD<float>(0.01, 0.9);  // learning_rate=0.01, momentum=0.9
    auto* optimizer2 = new nn::SGD<float>(0.01, 0.9);  // learning_rate=0.01, momentum=0.9 
    auto* optimizer3 = new nn::SGD<float>(0.01, 0.9);  // learning_rate=0.01, momentum=0.9 
    
    layer1->set_optimizer(optimizer1);
    layer2->set_optimizer(optimizer2);
    layer3->set_optimizer(optimizer3);
    
    try {
        std::cout << "Loading MNIST dataset..." << std::endl;
        
        std::string data_path = "./data/";  // adjust path if needed
        std::vector<Matrix<float> > training_images = mnist::load_images(data_path + "train-images.idx3-ubyte", 1000);
        std::vector<Matrix<float> > training_labels = mnist::load_labels(data_path + "train-labels.idx1-ubyte", 1000);
        
        std::vector<Matrix<float> > test_images = mnist::load_images(data_path + "t10k-images.idx3-ubyte", 100);
        std::vector<Matrix<float> > test_labels = mnist::load_labels(data_path + "t10k-labels.idx1-ubyte", 100);
        
        std::cout << "Loaded " << training_images.size() << " training images and " 
                  << test_images.size() << " test images." << std::endl;
        
        // Train
        network.set_verbosity(nn::Verbosity::DETAILED);
        std::cout << "\nTraining network...\n" << std::endl;
        network.train(training_images, training_labels, 10, 32);  // 10 epochs, batch size 32
        
        std::cout << "\nEvaluating on test set...\n" << std::endl;
        size_t correct = 0;
        for (size_t i = 0; i < test_images.size(); ++i) {
            Matrix<float> prediction = network.forward(test_images[i]);
            if (network.is_prediction_correct(prediction, test_labels[i])) {
                correct++;
            }
            
            // Visualize some predictions
            if (i < 10) {
             mnist::visualize_prediction(test_images[i], prediction, test_labels[i]);
            }
        }
        
        std::cout << "Test accuracy: " << (float)correct / test_images.size() * 100 << "%" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Make sure you have downloaded the MNIST dataset files and placed them in the data directory." << std::endl;
        std::cerr << "You can download them from: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/" << std::endl;
    }
    
    delete layer1;
    delete layer2;
    delete layer3;
    delete optimizer1;
    delete optimizer2;
    delete optimizer3;
    
    return 0;
}
