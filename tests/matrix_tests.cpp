#include <gtest/gtest.h>
#include "nn/matrix.hpp"

class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(MatrixTest, ConstructorTest) {
    // Basic constructor
    Matrix<float> m1(2, 2);
    EXPECT_EQ(m1.rows(), 2);
    EXPECT_EQ(m1.columns(), 2);

    // Constructor with values
    std::vector<float> values = {1.0, 2.0, 3.0, 4.0};
    Matrix<float> m2(2, 2, values);
    EXPECT_EQ(m2.at(0,0), 1.0);
    EXPECT_EQ(m2.at(0,1), 2.0);
    EXPECT_EQ(m2.at(1,0), 3.0);
    EXPECT_EQ(m2.at(1,1), 4.0);
}

TEST_F(MatrixTest, AdditionTest) {
    std::vector<float> values1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> values2 = {2.0, 3.0, 4.0, 5.0};
    Matrix<float> m1(2, 2, values1);
    Matrix<float> m2(2, 2, values2);
    
    Matrix<float> result = m1 + m2;
    
    EXPECT_EQ(result.at(0,0), 3.0);
    EXPECT_EQ(result.at(0,1), 5.0);
    EXPECT_EQ(result.at(1,0), 7.0);
    EXPECT_EQ(result.at(1,1), 9.0);
}

TEST_F(MatrixTest, MultiplicationTest) {
    std::vector<float> values1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> values2 = {2.0, 3.0, 4.0, 5.0};
    Matrix<float> m1(2, 2, values1);
    Matrix<float> m2(2, 2, values2);
    
    Matrix<float> result = m1 * m2;
    
    EXPECT_EQ(result.at(0,0), 10.0);
    EXPECT_EQ(result.at(0,1), 13.0);
    EXPECT_EQ(result.at(1,0), 22.0);
    EXPECT_EQ(result.at(1,1), 29.0);
}

TEST_F(MatrixTest, ScalarMultiplicationTest) {
    std::vector<float> values = {1.0, 2.0, 3.0, 4.0};
    Matrix<float> m(2, 2, values);
    
    Matrix<float> result = m * 2.0f;
    
    EXPECT_EQ(result.at(0,0), 2.0);
    EXPECT_EQ(result.at(0,1), 4.0);
    EXPECT_EQ(result.at(1,0), 6.0);
    EXPECT_EQ(result.at(1,1), 8.0);
}

TEST_F(MatrixTest, InvalidDimensionsTest) {
    Matrix<float> m1(2, 3);
    Matrix<float> m2(2, 2);
    
    EXPECT_THROW(m1 * m2, std::invalid_argument);
}

TEST_F(MatrixTest, HadamardProduct) {
    std::vector<float> values1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> values2 = {5.0, 6.0, 7.0, 8.0};

    Matrix<float> m1(2, 2, values1);
    Matrix<float> m2(2, 2, values2);

    Matrix<float> result = m1.hadamard(m2);

    EXPECT_EQ(result.at(0, 0), 5.0f);   // 1.0 * 5.0
    EXPECT_EQ(result.at(0, 1), 12.0f);  // 2.0 * 6.0
    EXPECT_EQ(result.at(1, 0), 21.0f);  // 3.0 * 7.0
    EXPECT_EQ(result.at(1, 1), 32.0f);  // 4.0 * 8.0
}

TEST_F(MatrixTest, HadamardProductDimensionMismatch) {
    Matrix<float> m1(2, 2);
    Matrix<float> m2(2, 3);

    EXPECT_THROW(m1.hadamard(m2), std::invalid_argument);
}
