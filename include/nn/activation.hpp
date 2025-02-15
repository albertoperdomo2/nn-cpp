 /* 
  * Activation Functions
  *
  * ReLU: Rectified Linear Unit
  * Function: f(x) = max(0, x)
  * Derivative: f'(x) = 1 if x > 0, 0 otherwise
  * Use cases: Hidden layers (most common choice)
  * Properties: Simple computation, no vanishing gradients, sparse activation (some neurons output exactly 0)
  *
  * Sigmoid
  * Function: f(x) = 1/(1 + e^(-x))
  * Derivative: f'(x) = f(x)(1 - f(x))
  * Use cases: Binary classification (output layer) or when outputs need to be interpreted as probabilities
  * Properties: Outputs between 0 and 1, smooth gradient, gradients can vanish
  *
  * Tanh
  * Function: f(x) = (e^x - e^(-x))/(e^x + e^(-x))
  * Derivative: f'(x) = 1 - tanh^2(x)
  * Use cases: When you need outputs between -1 and 1 or hidden layers
  * Properties: Zero-centered outputs
  *
  * LeakyReLU
  * Function: f(x) = x if x > 0, αx otherwise (α is small, like 0.01)
  * Derivative: f'(x) = 1 if x > 0, α otherwise
  * Use cases: Alternative to ReLU to prevent "dying ReLU" problem
  * Properties: Never completely "dies" (always has a small gradient)
 */

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <algorithm>
#include <cmath>

namespace nn {
 namespace activations {
  template<typename T>
  class ReLU {
   public:
    static T forward(const T x) {
     return std::max(static_cast<T>(0), x);
    }

    static T backward(const T x) {
     return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);
    }
  };

  template<typename T>
  class Sigmoid {
   public:
    static T forward(const T x) {
     // handle extreme values to prevent overflow/underflow
     // 100.0 is arbitrary
     if (x >= static_cast<T>(100.0)) return static_cast<T>(1);
     if (x <= static_cast<T>(-100.0)) return static_cast<T>(0);

     return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    }

    static T backward(const T x) {
     T fx = forward(x);
     return fx * (static_cast<T>(1) - fx);
    }
  };
  
  template<typename T>
  class Tanh {
   public:
    static T forward(const T x) {
     // handle extreme values to prevent overflow/underflow
     // 100.0 is arbitrary
     if (x >= static_cast<T>(100.0)) return static_cast<T>(1);
     if (x <= static_cast<T>(-100.0)) return static_cast<T>(-1);

     return std::tanh(x);
    }

    static T backward(const T x) {
     // handle extreme values to prevent overflow/underflow
     // 100.0 is arbitrary
     if (x >= static_cast<T>(100.0)) return static_cast<T>(0);
     if (x <= static_cast<T>(-100.0)) return static_cast<T>(0);

     return static_cast<T>(1) - std::pow(std::tanh(x), static_cast<T>(2));
    }
  };

  template<typename T>
  class LeakyReLU {
   public:
    static T forward(const T x, const T alpha = static_cast<T>(0.01)) {
     if (x > static_cast<T>(0)) {
      return x;
     }
     return x * alpha;
    }

    static T backward(const T x, const T alpha = static_cast<T>(0.01)) {
     return x > static_cast<T>(0) ? static_cast<T>(1) : alpha;
    }
  };
 }
}

#endif
