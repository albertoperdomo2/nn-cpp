#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.hpp"

namespace nn {

 template<typename T>
 class Optimizer {
  public:
   explicit Optimizer(T learning_rate) : learning_rate_(learning_rate) {}
   virtual ~Optimizer() = default;

   // pure virtual update method that derived classes must implement
   virtual void update(Matrix<T>& weights, 
     Matrix<T>& bias,
     const Matrix<T>& weight_gradients,
     const Matrix<T>& bias_gradients) = 0;

   // getter and setter
   T learning_rate() const { return learning_rate_; }
   void set_learning_rate(T lr) { learning_rate_ = lr; }

  protected:
   T learning_rate_;
 };

 template<typename T>
 class SGD: public Optimizer<T> {
  public:
   explicit SGD(T learning_rate, T momentum = 0.0)
    : Optimizer<T>(learning_rate), 
    momentum_(momentum),
    weight_velocity_(0, 0),  // will be resized on first update
    bias_velocity_(0, 0) {}

   void update(Matrix<T>& weights,
     Matrix<T>& bias,
     const Matrix<T>& weight_gradients,
     const Matrix<T>& bias_gradients) override {

    // init velocities in the first update
    if (weight_velocity_.rows() == 0) {
     weight_velocity_ = Matrix<T>(weights.rows(), weights.columns());
     bias_velocity_ = Matrix<T>(bias.rows(), bias.columns());
    }

    // update with momentum
    for (size_t i = 0; i < weights.rows(); i++) {
     for (size_t j = 0; j < weights.columns(); j++) {
      // v = momentum * v - learning_rate * gradient
      weight_velocity_.at(i, j) = 
       momentum_ * weight_velocity_.at(i, j) - 
       this->learning_rate_ * weight_gradients.at(i, j);

      // w = w + v
      weights.at(i, j) += weight_velocity_.at(i, j);
     }
    }

    // update biases
    for (size_t i = 0; i < bias.rows(); i++) {
     bias_velocity_.at(i, 0) = 
      momentum_ * bias_velocity_.at(i, 0) - 
      this->learning_rate_ * bias_gradients.at(i, 0);

     bias.at(i, 0) += bias_velocity_.at(i, 0);
    }
   }

  private:
   T momentum_;
   Matrix<T> weight_velocity_; // for momentum
   Matrix<T> bias_velocity_;

 };
}

#endif

