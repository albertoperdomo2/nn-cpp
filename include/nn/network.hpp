#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "layer.hpp"

namespace nn {

 enum class Verbosity {
  SILENT,     // no output
  MINIMAL,    // just epoch summaries
  DETAILED    // show more metrics
 };

 template<typename T>
 class Network {
  /*
   * This is a very simple approach in which users create layer objects and add them to the network.
   * I might refactor this in the future, or soon, if it fails to work.
   * Other approach would be to have a base layer class and build all layers from that one, 
   * without needing the user to create them manually.
   */
  private:
   std::vector<LayerBase<T>*> layers_;
   Verbosity verbosity_ = Verbosity::MINIMAL;

  public:
   Network() = default;
   ~Network() = default; // user responsible for layer cleanup

   void set_verbosity(Verbosity level) {
     verbosity_ = level;
   }

   // Add (an existing) layer to the network
   void add(LayerBase<T>* layer) {
    layers_.push_back(layer);
   }

   // Forward pass through all layers
   Matrix<T> forward(const Matrix<T>& input) {
    if (layers_.empty()) {
     throw std::runtime_error("network has no layers");
    }

    Matrix<T> current_output = input;

    for (auto& layer : layers_) {
     current_output = layer->forward(current_output);
    }

    return current_output;
   }

   // Backward pass through all layers
   void backward(const Matrix<T>& target, const Matrix<T>& output) {
    if (layers_.empty()) {
     throw std::runtime_error("network has no layers");
    }

    // Calculate initial error gradient based on the loss function
    Matrix<T> gradient = output - target; // for MSE loss

    // Backprpagate through layers in reverse order
    for (int i = layers_.size() - 1; i >= 0; --i) {
     gradient = layers_[i]->backward(gradient);
    }
   }

   Matrix<T> train_step(const Matrix<T>& input, const Matrix<T>& target) {
    Matrix<T> output = forward(input);
    backward(target, output);
    return output;
   }

   void train(const std::vector<Matrix<T>>& inputs,
     const std::vector<Matrix<T>>& targets,
     size_t epochs, size_t batch_size = 1) {
    /*
     * As one can tell, this training loop is aimed for a classification task.
     */

    if (inputs.size() != targets.size()) {
     throw std::invalid_argument("number of inputs must match number of targets");
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
     T total_loss = 0;
     size_t correct_predictions = 0;

     // Training loop with batch support
     for (size_t i = 0; i < inputs.size(); i += batch_size) {
      size_t current_batch_size = std::min(batch_size, inputs.size() - i);

      // Process one batch
      for (size_t j = 0; j < current_batch_size; ++j) {
       Matrix<T> output = train_step(inputs[i+j], targets[i+j]);

       T sample_loss = calculate_loss(output, targets[i+j]);
       total_loss += sample_loss;

       if (is_prediction_correct(output, targets[i+j])) {
        correct_predictions++;
       }
      }

      if (verbosity_ == Verbosity::DETAILED && (i/batch_size) % 10 == 0) {
       std::cout << "Epoch " << epoch+1 << ", Batch " << i/batch_size
        << ", Loss: " << total_loss/(i+current_batch_size) << std::endl;
      }
     }

     if (verbosity_ >= Verbosity::MINIMAL) {
      T avg_loss = total_loss / inputs.size();
      float accuracy = static_cast<float>(correct_predictions) / inputs.size() * 100;

      std::cout << "Epoch " << epoch+1 << "/" << epochs
       << ", Loss: " << avg_loss
       << ", Accuracy: " << accuracy << "%" << std::endl;
     }
    }
   }

   // Public for testing
   T calculate_loss(const Matrix<T>& output, const Matrix<T>& target) {
    // MSE
    T sum_squared_error = 0;
    for (size_t i = 0; i < output.rows(); ++i) {
     T error = output.at(i, 0) - target.at(i, 0);
     sum_squared_error += error * error;
    }
    return sum_squared_error / output.rows();
   }

   // Public for testing
   bool is_prediction_correct(const Matrix<T>& output, const Matrix<T>& target) {
    size_t predicted_class = 0;
    T max_value = output.at(0, 0);

    for (size_t i = 1; i < output.rows(); ++i) {
     if (output.at(i, 0) > max_value) { // check if highest probability matches target
      max_value = output.at(i, 0);
      predicted_class = i;
     }
    }

    size_t target_class = 0;
    T target_max = target.at(0, 0);

    for (size_t i = 1; i < target.rows(); ++i) {
     if (target.at(i, 0) > target_max) {
      target_max = target.at(i, 0);
      target_class = i;
     }
    }

    return predicted_class == target_class;
   }
 };
}

#endif
