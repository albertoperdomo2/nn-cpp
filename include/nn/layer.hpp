#ifndef LAYER_H
#define LAYER_H

#include <cmath>
#include <random>
#include "matrix.hpp"
#include "activation.hpp"
#include "optimizer.hpp"

namespace nn {

 enum class InitializationType {
  XAVIER_UNIFORM,  // Uniform distribution (good for tanh/sigmoid)
  XAVIER_NORMAL,   // Normal distribution (good for tanh/sigmoid)
  HE_UNIFORM,      // Uniform distribution (good for ReLU)
  HE_NORMAL,       // Normal distribution (good for ReLU)
  ZERO             // All weights = 0 (for testing)
 };

 template<typename T, template<typename> class Activation>
 class Layer {
  private:
   Matrix<T> weights_;
   Matrix<T> bias_;
   size_t input_size_;
   size_t output_size_;
   T learning_rate_;
   std::mt19937 gen_;

   Matrix<T> last_input_;       // Store input for backward pass
   Matrix<T> last_z_;           // Store weighted sum (before activation)
   Matrix<T> last_activation_;  // Store output after activation
   
   Optimizer<T>* optimizer_ = nullptr;

   void initialize_weights(InitializationType type) {
    switch(type) {
     case InitializationType::XAVIER_UNIFORM: {
         T x = std::sqrt(6.0 / (input_size_ + output_size_));
         std::uniform_real_distribution<T> dist(-x, x);
         fill_weights(dist);
         break;
     }
     case InitializationType::XAVIER_NORMAL: {
         T std = std::sqrt(2.0 / (input_size_ + output_size_));
         std::normal_distribution<T> dist(0.0, std); // mean 0.0
         fill_weights(dist);
         break;
     }
     case InitializationType::HE_UNIFORM: {
         T x = std::sqrt(6.0 / input_size_);
         std::uniform_real_distribution<T> dist(-x, x);
         fill_weights(dist);
         break;
     }
     case InitializationType::HE_NORMAL: {
         T std = std::sqrt(2.0 / input_size_);
         std::normal_distribution<T> dist(0.0, std); // mean 0.0
         fill_weights(dist);
         break;
     }
     case InitializationType::ZERO:
     default: {
         for (size_t i = 0; i < weights_.rows(); i++) {
          for (size_t j = 0; j < weights_.columns(); j++) {
           weights_.at(i, j) = 0;
          }
         }
         break;
     }
    }

    // Initialize biases to zero
    for (size_t i = 0; i < bias_.rows(); i++) {
     bias_.at(i, 0) = 0;
    }
   }

   template<typename Distribution>
   void fill_weights(Distribution& dist) {
    for (size_t i = 0; i < weights_.rows(); i++) {
     for (size_t j = 0; j < weights_.columns(); j++) {
      weights_.at(i, j) = dist(gen_);
     }
    }
   }

   void update_parameters(const Matrix<T>& weight_gradients,
                         const Matrix<T>& bias_gradients) {
    for (size_t i = 0; i < weights_.rows(); i++) {
     for (size_t j = 0; j < weights_.columns(); j++) {
      weights_.at(i, j) -= learning_rate_ * weight_gradients.at(i, j);
     }
    }

    for (size_t i = 0; i < bias_.rows(); i++) {
     bias_.at(i, 0) -= learning_rate_ * bias_gradients.at(i, 0);
    }
   }

  public:
   Layer(size_t input_size, size_t output_size, T learning_rate = 0.01,
     InitializationType init_type = InitializationType::XAVIER_UNIFORM)
    : input_size_(input_size),
      output_size_(output_size),
      weights_(output_size, input_size),
      bias_(output_size, 1),
      learning_rate_(learning_rate),
      last_input_(input_size, 1),
      last_z_(output_size, 1),
      last_activation_(output_size, 1)
   {
    /*
     * Default: Glorot or Xavier uniform initialization
     * Principles:
     * - The variance of weights should be inversely proportional to the square root of the number of inputs and outputs.
     * - The distribution is centered at zero, which prevents the mean of activations from shifting too far from zero as signals propagate forward.
     * - Using a uniform distribution bounded by the calculated limits ensures weights stay within a reasonable range while still allowing for sufficient variation.
     *
     * This initialization scheme works particularly well with tanh activations, which was the standard when Glorot introduced it. For ReLU activations, a modified version called He initialization is often preferred.
     *
    */

    std::random_device rd;
    gen_ = std::mt19937(rd());

    initialize_weights(init_type);
   }
   
   void set_optimizer(Optimizer<T>* optimizer) {
    optimizer_ = optimizer;
   }

   // For testing
   void set_weights(Matrix<T> weights) {
    weights_ = weights;
   }
    
   // For testing
   void set_bias(Matrix<T> bias) {
    bias_ = bias;
   }

   const Matrix<T>& weights() const { return weights_; }
   const Matrix<T>& bias() const { return bias_; }

   Matrix<T> forward(const Matrix<T>& input) {
    if (input.columns() != 1 || input.rows() != input_size_) {
     throw std::invalid_argument("input dimensions do not match layer input size");
    }

    last_input_ = input;
    last_z_ = weights_ * input + bias_;

    Matrix<T> output(output_size_, 1);
    for (size_t i = 0; i < output_size_; i++) {
     output.at(i, 0) = Activation<T>::forward(last_z_.at(i, 0));
    }

    last_activation_ = output;

    return output;
   }

   Matrix<T> backward(const Matrix<T>& gradient_from_next_layer) {
    Matrix<T> activation_gradient(output_size_, 1);
    for (size_t i = 0; i < output_size_; i++) {
     activation_gradient.at(i, 0) = Activation<T>::backward(last_z_.at(i, 0));
    }

    Matrix<T> delta = gradient_from_next_layer.hadamard(activation_gradient);
    Matrix<T> weight_gradients = delta * last_input_.transpose();
    Matrix<T> input_gradients = weights_.transpose() * delta;

    if (optimizer_) {
     optimizer_->update(weights_, bias_, weight_gradients, delta);
    }

    return input_gradients;
   }
 };
}

#endif
