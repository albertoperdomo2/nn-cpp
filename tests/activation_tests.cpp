#include <gtest/gtest.h>
#include "nn/activation.hpp"

class ReLUTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ReLUTest, ForwardPass) {
    // Test positive input
    EXPECT_FLOAT_EQ(nn::activations::ReLU<float>::forward(2.0f), 2.0f);

    // Test negative input
    EXPECT_FLOAT_EQ(nn::activations::ReLU<float>::forward(-2.0f), 0.0f);

    // Test zero input
    EXPECT_FLOAT_EQ(nn::activations::ReLU<float>::forward(0.0f), 0.0f);
}

TEST_F(ReLUTest, BackwardPass) {
    // Test positive input
    EXPECT_FLOAT_EQ(nn::activations::ReLU<float>::backward(2.0f), 1.0f);

    // Test negative input
    EXPECT_FLOAT_EQ(nn::activations::ReLU<float>::backward(-2.0f), 0.0f);

    // Test zero input
    EXPECT_FLOAT_EQ(nn::activations::ReLU<float>::backward(0.0f), 0.0f);
}

class SigmoidTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SigmoidTest, ForwardPass) {
    // Test positive input
    EXPECT_FLOAT_EQ(nn::activations::Sigmoid<float>::forward(2.0f), 0.880797077978f);

    // Test negative input
    EXPECT_FLOAT_EQ(nn::activations::Sigmoid<float>::forward(-2.0f), 0.119202922022f);

    // Test zero input
    EXPECT_FLOAT_EQ(nn::activations::Sigmoid<float>::forward(0.0f), 0.5f);

    // Test extreme values
    EXPECT_FLOAT_EQ(nn::activations::Sigmoid<float>::forward(100.0f), 1.0f);
    EXPECT_FLOAT_EQ(nn::activations::Sigmoid<float>::forward(-100.0f), 0.0f);
}

TEST_F(SigmoidTest, BackwardPass) {
    // Test positive input
    float fx = nn::activations::Sigmoid<float>::forward(2.0f);
    EXPECT_FLOAT_EQ(nn::activations::Sigmoid<float>::backward(2.0f), fx * (1.0f - fx));

    // Test negative input
    fx = nn::activations::Sigmoid<float>::forward(-2.0f);
    EXPECT_FLOAT_EQ(nn::activations::Sigmoid<float>::backward(-2.0f), fx * (1.0f - fx));

    // Test zero input
    EXPECT_FLOAT_EQ(nn::activations::Sigmoid<float>::backward(0.0f), 0.25f);
}

TEST_F(SigmoidTest, Properties) {
    // Test that output is always between 0 and 1
    for (float x = -10.0f; x <= 10.0f; x += 0.5f) {
     float y = nn::activations::Sigmoid<float>::forward(x);
     EXPECT_GE(y, 0.0f);
     EXPECT_LE(y, 1.0f);
    }

    // Test that derivative is always positive and maximum at x=0
    float max_derivative = nn::activations::Sigmoid<float>::backward(0.0f);
    for (float x = -10.0f; x <= 10.0f; x += 0.5f) {
     if (x == 0.0f) continue;
     float derivative = nn::activations::Sigmoid<float>::backward(x);
     EXPECT_GE(derivative, 0.0f);
     EXPECT_LT(derivative, max_derivative);
    }
}

class TanhTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TanhTest, ForwardPass) {
    // Test positive input
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::forward(2.0f), 0.964027580075f);

    // Test negative input
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::forward(-2.0f), -0.964027580075f);

    // Test zero input
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::forward(0.0f), 0.0f);

    // Test extreme values
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::forward(100.0f), 1.0f);
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::forward(-100.0f), -1.0f);
}

TEST_F(TanhTest, BackwardPass) {
    // Test positive input
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::backward(2.0f), 0.070650824831f);

    // Test negative input
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::backward(-2.0f), 0.070650824831f);

    // Test zero input
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::backward(0.0f), 1.0f);

    // Test extreme values
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::backward(100.0f), 0.0f);
    EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::backward(-100.0f), 0.0f);
}

TEST_F(TanhTest, Properties) {
    // Test that output is always between 0 and 1
    for (float x = -10.0f; x <= 10.0f; x += 0.5f) {
     float y = nn::activations::Tanh<float>::forward(x);
     EXPECT_GE(y, -1.0f);
     EXPECT_LE(y, 1.0f);
    }

    // Test that tanh is symmetric
    for (float x = -10.0f; x <= 10.0f; x += 0.5f) {
     EXPECT_FLOAT_EQ(nn::activations::Tanh<float>::forward(-x), 
       -nn::activations::Tanh<float>::forward(x));
    }

    // Test that derivative is always positive and maximum at x=0
    float max_derivative = nn::activations::Tanh<float>::backward(0.0f);
    for (float x = -10.0f; x <= 10.0f; x += 0.5f) {
     if (x == 0.0f) continue;
     float derivative = nn::activations::Tanh<float>::backward(x);
     EXPECT_GE(derivative, 0.0f);
     EXPECT_LT(derivative, max_derivative);
    }
}

class LeakyReLUTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LeakyReLUTest, ForwardPass) {
    // Test positive input
    EXPECT_FLOAT_EQ(nn::activations::LeakyReLU<float>::forward(2.0f, 0.01f), 2.0f);

    // Test negative input
    EXPECT_FLOAT_EQ(nn::activations::LeakyReLU<float>::forward(-2.0f, 0.01f), -0.02f);

    // Test zero input
    EXPECT_FLOAT_EQ(nn::activations::LeakyReLU<float>::forward(0.0f, 0.01f), 0.0f);
}

TEST_F(LeakyReLUTest, BackwardPass) {
    // Test positive input
    EXPECT_FLOAT_EQ(nn::activations::LeakyReLU<float>::backward(2.0f, 0.01f), 1.0f);

    // Test negative input
    EXPECT_FLOAT_EQ(nn::activations::LeakyReLU<float>::backward(-2.0f, 0.01f), 0.01f);

    // Test zero input
    EXPECT_FLOAT_EQ(nn::activations::LeakyReLU<float>::backward(0.0f, 0.01f), 0.01f);
}
