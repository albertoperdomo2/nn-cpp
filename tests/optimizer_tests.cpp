#include <gtest/gtest.h>
#include "nn/optimizer.hpp"

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(OptimizerTest, SGDLearningRate) {
    // Instantiate optimizer
    auto* optimizer = new nn::SGD<float>(0.01f, 0.9f); // learning_rate=0.01, momentum=0.9
    
    // Learning rate getter
    EXPECT_FLOAT_EQ(optimizer->learning_rate(), 0.01f);
    
    // Learning rate setter
    optimizer->set_learning_rate(0.001f);
    EXPECT_FLOAT_EQ(optimizer->learning_rate(), 0.001f);

    delete optimizer;
}

TEST_F(OptimizerTest, SGDWithoutMomentum) {
    // Create initial weights and biases
    std::vector<float> w_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_values = {7.0f, 8.0f, 9.0f};
    
    // Create gradient matrices
    std::vector<float> w_grad_values = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    std::vector<float> b_grad_values = {0.7f, 0.8f, 0.9f};
    
    // Setup matrices
    Matrix<float> weights(3, 2, w_values);
    Matrix<float> biases(3, 1, b_values);
    Matrix<float> weight_gradients(3, 2, w_grad_values);
    Matrix<float> bias_gradients(3, 1, b_grad_values);
    
    // Expected values after update with learning_rate=0.01 and momentum=0.0
    std::vector<float> expected_w = {
        1.0f - 0.01f * 0.1f,  // 0.999
        2.0f - 0.01f * 0.2f,  // 1.998
        3.0f - 0.01f * 0.3f,  // 2.997
        4.0f - 0.01f * 0.4f,  // 3.996
        5.0f - 0.01f * 0.5f,  // 4.995
        6.0f - 0.01f * 0.6f   // 5.994
    };
    std::vector<float> expected_b = {
        7.0f - 0.01f * 0.7f,  // 6.993
        8.0f - 0.01f * 0.8f,  // 7.992
        9.0f - 0.01f * 0.9f   // 8.991
    };
    
    Matrix<float> expected_weights(3, 2, expected_w);
    Matrix<float> expected_biases(3, 1, expected_b);
    
    // Instantiate optimizer without momentum
    auto* optimizer = new nn::SGD<float>(0.01f, 0.0f);
    
    // Perform update
    optimizer->update(weights, biases, weight_gradients, bias_gradients);
    
    // Check if weights and biases were updated correctly
    for (size_t i = 0; i < weights.rows(); i++) {
        for (size_t j = 0; j < weights.columns(); j++) {
            EXPECT_NEAR(weights.at(i, j), expected_weights.at(i, j), 1e-5f);
        }
    }
    
    for (size_t i = 0; i < biases.rows(); i++) {
        EXPECT_NEAR(biases.at(i, 0), expected_biases.at(i, 0), 1e-5f);
    }
    
    delete optimizer;
}

TEST_F(OptimizerTest, SGDWithMomentum) {
    // Create initial weights and biases
    std::vector<float> w_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_values = {7.0f, 8.0f, 9.0f};
    
    // Create gradient matrices
    std::vector<float> w_grad_values = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    std::vector<float> b_grad_values = {0.7f, 0.8f, 0.9f};
    
    // Setup matrices
    Matrix<float> weights(3, 2, w_values);
    Matrix<float> biases(3, 1, b_values);
    Matrix<float> weight_gradients(3, 2, w_grad_values);
    Matrix<float> bias_gradients(3, 1, b_grad_values);
    
    // Instantiate optimizer with momentum=0.9
    auto* optimizer = new nn::SGD<float>(0.01f, 0.9f);
    
    // First update: velocities are initialized to zero
    // v_w = 0*0 - 0.01*grad = -0.01*grad
    // w = w + v_w = w - 0.01*grad
    optimizer->update(weights, biases, weight_gradients, bias_gradients);
    
    // Expected values after first update
    std::vector<float> expected_w_first = {
        1.0f - 0.01f * 0.1f,  // 0.999
        2.0f - 0.01f * 0.2f,  // 1.998
        3.0f - 0.01f * 0.3f,  // 2.997
        4.0f - 0.01f * 0.4f,  // 3.996
        5.0f - 0.01f * 0.5f,  // 4.995
        6.0f - 0.01f * 0.6f   // 5.994
    };
    std::vector<float> expected_b_first = {
        7.0f - 0.01f * 0.7f,  // 6.993
        8.0f - 0.01f * 0.8f,  // 7.992
        9.0f - 0.01f * 0.9f   // 8.991
    };
    
    Matrix<float> expected_weights_first(3, 2, expected_w_first);
    Matrix<float> expected_biases_first(3, 1, expected_b_first);
    
    // Check if weights and biases were updated correctly after first update
    for (size_t i = 0; i < weights.rows(); i++) {
        for (size_t j = 0; j < weights.columns(); j++) {
            EXPECT_NEAR(weights.at(i, j), expected_weights_first.at(i, j), 1e-5f);
        }
    }
    
    for (size_t i = 0; i < biases.rows(); i++) {
        EXPECT_NEAR(biases.at(i, 0), expected_biases_first.at(i, 0), 1e-5f);
    }
    
    // Second update: with momentum
    // v_w = 0.9*v_w_prev - 0.01*grad
    //     = 0.9*(-0.01*grad) - 0.01*grad
    //     = -0.009*grad - 0.01*grad
    //     = -0.019*grad
    // w = w + v_w = w_prev - 0.019*grad
    optimizer->update(weights, biases, weight_gradients, bias_gradients);
    
    // Expected values after second update
    std::vector<float> expected_w_second = {
        expected_w_first[0] + (-0.009f * 0.1f - 0.01f * 0.1f),
        expected_w_first[1] + (-0.009f * 0.2f - 0.01f * 0.2f),
        expected_w_first[2] + (-0.009f * 0.3f - 0.01f * 0.3f),
        expected_w_first[3] + (-0.009f * 0.4f - 0.01f * 0.4f),
        expected_w_first[4] + (-0.009f * 0.5f - 0.01f * 0.5f),
        expected_w_first[5] + (-0.009f * 0.6f - 0.01f * 0.6f)
    };
    std::vector<float> expected_b_second = {
        expected_b_first[0] + (-0.009f * 0.7f - 0.01f * 0.7f),
        expected_b_first[1] + (-0.009f * 0.8f - 0.01f * 0.8f),
        expected_b_first[2] + (-0.009f * 0.9f - 0.01f * 0.9f)
    };
    
    Matrix<float> expected_weights_second(3, 2, expected_w_second);
    Matrix<float> expected_biases_second(3, 1, expected_b_second);
    
    // Check if weights and biases were updated correctly after second update
    for (size_t i = 0; i < weights.rows(); i++) {
        for (size_t j = 0; j < weights.columns(); j++) {
            EXPECT_NEAR(weights.at(i, j), expected_weights_second.at(i, j), 1e-5f);
        }
    }
    
    for (size_t i = 0; i < biases.rows(); i++) {
        EXPECT_NEAR(biases.at(i, 0), expected_biases_second.at(i, 0), 1e-5f);
    }
    
    delete optimizer;
}
