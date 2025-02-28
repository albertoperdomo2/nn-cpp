#include <gtest/gtest.h>
#include "nn/layer.hpp"
#include "nn/optimizer.hpp"

class LayerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LayerTest, ConstructorTest) {
    nn::Layer<float, nn::activations::ReLU> layer(5, 3);
    EXPECT_EQ(layer.weights().rows(), 3);
    EXPECT_EQ(layer.weights().columns(), 5);
    EXPECT_EQ(layer.bias().rows(), 3);
    EXPECT_EQ(layer.bias().columns(), 1);
}

TEST_F(LayerTest, ForwardPass) {
    // Create a 2 input, 2 output, ReLU layer
    nn::Layer<float, nn::activations::ReLU> layer(2, 2);

    // Create initial weights and biases
    std::vector<float> w_values = {0.5f, 0.8f, 0.1f, 0.2f};
    std::vector<float> b_values = {0.1f, 0.2f};
    
    Matrix<float> weights(2, 2, w_values);
    Matrix<float> bias(2, 1, b_values);

    // Manually set weight and biases
    layer.set_weights(weights);
    layer.set_bias(bias);

    // Create input
    std::vector<float> input_values = {0.5f, 1.0f};
    Matrix<float> input(2, 1, input_values);

    // Perform forward pass
    Matrix<float> output = layer.forward(input);

    //  Calculate weighted sum: z = Wx + b
    //  [0.5, 0.8] * [0.5] + [0.1] = [0.5x0.5 + 0.8x1.0 + 0.1] = [1.15]
    //  [0.1, 0.2] * [1.0] + [0.2]   [0.1x0.5 + 0.2x1.0 + 0.2]   [0.45]
    //  Apply ReLU:
    //  ReLU(1.15) = 1.15
    //  ReLU(0.45) = 0.45
    EXPECT_NEAR(output.at(0, 0), 1.15f, 0.001f);
    EXPECT_NEAR(output.at(1, 0), 0.45f, 0.001f);
}

TEST_F(LayerTest, BackwardPass) {
    // Create a 2 input, 2 output, ReLU layer
    nn::Layer<float, nn::activations::ReLU> layer(2, 2);

    // Create initial weights and biases
    std::vector<float> w_values = {0.5f, 0.8f, 0.1f, 0.2f};
    std::vector<float> b_values = {0.1f, 0.2f};
    
    Matrix<float> weights(2, 2, w_values);
    Matrix<float> bias(2, 1, b_values);

    // Manually set weight and biases
    layer.set_weights(weights);
    layer.set_bias(bias);

    // Create input
    std::vector<float> input_values = {0.5f, 1.0f};
    Matrix<float> input(2, 1, input_values);

    // Perform forward pass
    Matrix<float> output = layer.forward(input);

    // Verify forward pass worked correctly
    // For how I got these numbers, see ForwardPass test
    EXPECT_NEAR(output.at(0, 0), 1.15f, 0.001f);
    EXPECT_NEAR(output.at(1, 0), 0.45f, 0.001f);

    // Create gradient for backward pass
    // In a real network, this would come from the next layer
    std::vector<float> grad_values = {1.0f, 1.0f};
    Matrix<float> gradient(2, 1, grad_values);

    // Perform backward pass
    Matrix<float> input_gradient = layer.backward(gradient);

    // Check that input gradients are correct
    // Apply ReLU':
    // ReLU'(1.15) = 1
    // ReLU'(0.45) = 1
    // Element-wise product of incoming gradient and activation gradient is the same in both cases:
    // [1.0 x 1] = [1.0]
    // Weight gradients (delta x input_transpose) is the same in both cases:
    // [1.0] x [0.5, 1.0] = [0.5, 1.0]
    // Input gradients (weights_transpose x delta) are:
    // [0.5, 0.1] x [1.0] = [0.6]
    // [0.8, 0.2] × [1.0] = [1.0]
    EXPECT_NEAR(input_gradient.at(0, 0), 0.6f, 0.001f);
    EXPECT_NEAR(input_gradient.at(1, 0), 1.0f, 0.001f);
}

TEST_F(LayerTest, OptimizerIntegrationTest) {
    // Create a 2 input, 2 output, ReLU layer
    nn::Layer<float, nn::activations::ReLU> layer(2, 2);
    
    // Create initial weights and biases
    std::vector<float> w_values = {0.5f, 0.8f, 0.1f, 0.2f};
    std::vector<float> b_values = {0.1f, 0.2f};
    
    Matrix<float> weights(2, 2, w_values);
    Matrix<float> bias(2, 1, b_values);

    // Manually set weight and biases
    layer.set_weights(weights);
    layer.set_bias(bias);

    // Create and set optimizer
    nn::SGD<float>* optimizer = new nn::SGD<float>(0.1f); // learning_rate = 0.1
    layer.set_optimizer(optimizer);

    // Create input
    std::vector<float> input_values = {0.5f, 1.0f};
    Matrix<float> input(2, 1, input_values);

    // Perform forward pass
    Matrix<float> output = layer.forward(input);

    // Create gradient for backward pass
    // In a real network, this would come from the next layer
    std::vector<float> grad_values = {1.0f, 1.0f};
    Matrix<float> gradient(2, 1, grad_values);

    // Perform backward pass (this will update weights via optimizer)
    layer.backward(gradient);
    
    // Check that weights were updated correctly: original_weight - learning_rate * gradient
    EXPECT_NEAR(layer.weights().at(0, 0), 0.45f, 0.001f); // 0.5 - 0.1*0.5
    EXPECT_NEAR(layer.weights().at(0, 1), 0.7f, 0.001f);  // 0.8 - 0.1*1.0
    EXPECT_NEAR(layer.weights().at(1, 0), 0.05f, 0.001f); // 0.1 - 0.1*0.5
    EXPECT_NEAR(layer.weights().at(1, 1), 0.1f, 0.001f);  // 0.2 - 0.1*1.0
    
    // Check biases were updated
    EXPECT_NEAR(layer.bias().at(0, 0), 0.0f, 0.001f);  // 0.1 - 0.1*1.0
    EXPECT_NEAR(layer.bias().at(1, 0), 0.1f, 0.001f);  // 0.2 - 0.1*1.0
    
    delete optimizer;
}
