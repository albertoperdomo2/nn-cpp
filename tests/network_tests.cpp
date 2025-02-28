#include <gtest/gtest.h>
#include "nn/network.hpp"
#include "nn/layer.hpp"
#include "nn/optimizer.hpp"
#include "nn/activation.hpp"

class NetworkTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(NetworkTest, ConstructorTest) {
    // Constructor
    EXPECT_NO_THROW({
      nn::Network<float> test_network;
    });
}

TEST_F(NetworkTest, AddLayerTest) {
    nn::Network<float> network;
    
    auto* layer = new nn::Layer<float, nn::activations::ReLU>(5, 3);
    network.add(layer);
    
    // I don't have a direct way to check if the layer was added,
    // but if a forward pass succeeds, then it was added
    std::vector<float> input_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Matrix<float> input(5, 1, input_values);
    
    EXPECT_NO_THROW(network.forward(input));
    
    delete layer;
}

TEST_F(NetworkTest, ForwardPassTest) {
    nn::Network<float> network;
    
    // Create layers with known weights/bias for predictable output
    auto* layer1 = new nn::Layer<float, nn::activations::ReLU>(2, 2);
    auto* layer2 = new nn::Layer<float, nn::activations::Sigmoid>(2, 1);
    
    // Set specific weights for testing
    std::vector<float> w_values_1 = {0.5f, 0.8f, 0.1f, 0.2f};
    std::vector<float> w_values_2 = {0.3f, 0.6f};
    std::vector<float> b_values_1 = {0.1f, 0.2f};
    std::vector<float> b_values_2 = {0.1f};


    Matrix<float> w1(2, 2, w_values_1);
    Matrix<float> b1(2, 1, b_values_1);
    Matrix<float> w2(1, 2, w_values_2);
    Matrix<float> b2(1, 1, b_values_2);
    
    // Manually set weight and biases
    layer1->set_weights(w1);
    layer1->set_bias(b1);
    layer2->set_weights(w2);
    layer2->set_bias(b2);
    
    // Add layers to the network
    network.add(layer1);
    network.add(layer2);
    
    // Create input
    std::vector<float> input_values = {0.5f, 1.0f};
    Matrix<float> input(2, 1, input_values);
    
    // Perform forward pass
    Matrix<float> output = network.forward(input);
    
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.columns(), 1);
    
    // Calculate expected output:
    // Layer 1: [0.5*0.5+0.8*1.0+0.1, 0.1*0.5+0.2*1.0+0.2] = [1.15, 0.45]
    // After ReLU: [1.15, 0.45]
    // Layer 2: [0.3*1.15+0.6*0.45+0.1] = [0.645]
    // After Sigmoid: [1/(1+e^-0.645)] ≈ [0.656]
    
    EXPECT_NEAR(output.at(0, 0), 0.656, 0.02);
    
    delete layer1;
    delete layer2;
}

TEST_F(NetworkTest, XorTrainingTest) {
    nn::Network<float> network;
    
    // Create a 2-layer network for XOR
    auto* layer1 = new nn::Layer<float, nn::activations::ReLU>(2, 3);
    auto* layer2 = new nn::Layer<float, nn::activations::Sigmoid>(3, 1);
    
    network.add(layer1);
    network.add(layer2);
    
    auto* optimizer1 = new nn::SGD<float>(0.1); // learning_rate = 0.1
    auto* optimizer2 = new nn::SGD<float>(0.1); // learning_rate = 0.1
    
    layer1->set_optimizer(optimizer1);
    layer2->set_optimizer(optimizer2);
    
    // XOR training data
    std::vector<Matrix<float>> inputs;
    std::vector<Matrix<float>> targets;
    
    // Input: [0,0] -> Output: [0]
    std::vector<float> in1 = {0.0f, 0.0f};
    std::vector<float> out1 = {0.0f};
    inputs.push_back(Matrix<float>(2, 1, in1));
    targets.push_back(Matrix<float>(1, 1, out1));
    
    // Input: [0,1] -> Output: [1]
    std::vector<float> in2 = {0.0f, 1.0f};
    std::vector<float> out2 = {1.0f};
    inputs.push_back(Matrix<float>(2, 1, in2));
    targets.push_back(Matrix<float>(1, 1, out2));
    
    // Input: [1,0] -> Output: [1]
    std::vector<float> in3 = {1.0f, 0.0f};
    std::vector<float> out3 = {1.0f};
    inputs.push_back(Matrix<float>(2, 1, in3));
    targets.push_back(Matrix<float>(1, 1, out3));
    
    // Input: [1,1] -> Output: [0]
    std::vector<float> in4 = {1.0f, 1.0f};
    std::vector<float> out4 = {0.0f};
    inputs.push_back(Matrix<float>(2, 1, in4));
    targets.push_back(Matrix<float>(1, 1, out4));
    
    // Silent mode for testing
    network.set_verbosity(nn::Verbosity::SILENT);
    
    network.train(inputs, targets, 1000);
    
    // Test the network on the same inputs (not real case tho)
    float total_loss = 0.0f;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Matrix<float> output = network.forward(inputs[i]);
        total_loss += network.calculate_loss(output, targets[i]);
    }
    
    // Check if loss is low enough to indicate learning
    EXPECT_LT(total_loss / inputs.size(), 0.13);
    
    delete layer1;
    delete layer2;
    delete optimizer1;
    delete optimizer2;
}

TEST_F(NetworkTest, LossCalculationTest) {
    nn::Network<float> network;
    
    // Create simple output and target
    std::vector<float> output_values = {0.1f, 0.7f, 0.2f};
    std::vector<float> target_values = {0.0f, 1.0f, 0.0f};
    
    Matrix<float> output(3, 1, output_values);
    Matrix<float> target(3, 1, target_values);
    
    // Calculate MSE: (0.1-0)^2 + (0.7-1)^2 + (0.2-0)^2 / 3 = 0.01 + 0.09 + 0.04 / 3 = 0.14/3 = 0.0467
    EXPECT_NEAR(network.calculate_loss(output, target), 0.0467f, 0.001);
}

TEST_F(NetworkTest, PredictionCorrectnessTest) {
    nn::Network<float> network;
    
    // Correct prediction (highest value matches target)
    std::vector<float> output1 = {0.1f, 0.7f, 0.2f};
    std::vector<float> target1 = {0.0f, 1.0f, 0.0f};
    Matrix<float> out1(3, 1, output1);
    Matrix<float> tgt1(3, 1, target1);
    
    EXPECT_TRUE(network.is_prediction_correct(out1, tgt1));
    
    // Incorrect prediction
    std::vector<float> output2 = {0.1f, 0.2f, 0.7f};
    std::vector<float> target2 = {0.0f, 1.0f, 0.0f};
    Matrix<float> out2(3, 1, output2);
    Matrix<float> tgt2(3, 1, target2);
    
    EXPECT_FALSE(network.is_prediction_correct(out2, tgt2));
}
