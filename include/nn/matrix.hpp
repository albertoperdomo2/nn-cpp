#ifndef MATRIX_H
#define MATRIX_H

#include <iostream> 
#include <iterator>
#include <iomanip> 
#include <vector> 
#include <stdexcept>

template<typename T>
class Matrix {
 private:
  size_t rows_;
  size_t columns_;
  std::vector<T> data_;

  // Unsafe and unbound methods for internal usage
  inline T& unsafe_at(size_t i, size_t j) {
   return data_[i * columns_ + j];
  }

  inline const T& unsafe_at(size_t i, size_t j) const {
   return data_[i * columns_ + j];
  }

 public:
  Matrix(size_t rows, size_t columns) 
   : rows_(rows), columns_(columns), data_(rows * columns) {};

  Matrix(size_t rows, size_t columns, const std::vector<T>& values)
   : rows_(rows), columns_(columns), data_(values) {
   if (values.size() != rows * columns) {
    throw std::invalid_argument("initial values size doesn't match matrix dimensions");
   }
   }

  Matrix(const Matrix&) = default;
  Matrix& operator=(const Matrix&) = default;
  Matrix(Matrix&&) noexcept = default;
  Matrix& operator=(Matrix&&) noexcept = default;

  // Access element at (i,j)
  T& at(size_t i, size_t j) {
   if (i >= rows_ || j >= columns_) {
    throw std::out_of_range("matrix indices out of range");
   }
   return data_[i * columns_ + j];
  }

  const T& at(size_t i, size_t j) const {
   if (i >= rows_ || j >= columns_) {
    throw std::out_of_range("matrix indices out of range");
   }
   return data_[i * columns_ + j];
  }

  // Get dimensions
  size_t rows() const { return rows_; }
  size_t columns() const { return columns_; }

  // Matrix operations
  Matrix<T> add(const Matrix<T>& A) const {
   if (A.rows() != rows_ || A.columns() != columns_) {
    throw std::invalid_argument(std::string(__func__) + ": matrices are not the same size");
   }

   Matrix<T> result(rows_, columns_);

   for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < columns_; j++) {
     result.at(i, j) = A.at(i,j) + at(i,j);
    }
   }

   return result;
  }

  Matrix<T>& add_inplace(const Matrix<T>& A) {
   if (A.rows() != rows_ || A.columns() != columns_) {
    throw std::invalid_argument(std::string(__func__) + ": matrices are not the same size");
   }

   for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < columns_; j++) {
     unsafe_at(i, j) += A.at(i, j);
    }
   }

   return *this;
  }

  Matrix<T> operator+(const Matrix<T>& A) const { return add(A); }
  Matrix<T>& operator+=(const Matrix<T>& A) { return add_inplace(A); }

  Matrix<T> mul(const Matrix<T>& A) const {
   if (A.rows() != columns_) {
    throw std::invalid_argument(std::string(__func__) + ": matrices cannot be multiplied");
   }

   Matrix<T> result(rows_, A.columns());
   
   for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < A.columns(); j++) {
     T sum = 0;
     for (size_t k = 0; k < columns_; k++) {
      sum += at(i, k) * A.at(k, j);
     }
     result.at(i,j) = sum;
    }
   }

   return result;
  }

  Matrix<T> scalar_mul(const T scalar) const {
   Matrix<T> result(rows_, columns_);

   for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < columns_; j++) {
     result.at(i,j) = at(i,j) * scalar;
    }
   }

   return result;
  }

  Matrix<T>& scalar_mul_inplace(const T scalar) {
   for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < columns_; j++) {
     unsafe_at(i, j) *= scalar;
    }
   }

   return *this;
  }
   
  Matrix<T> operator*(const Matrix<T>& A) const { return mul(A); }
  Matrix<T> operator*(const T scalar) const { return scalar_mul(scalar); }
  Matrix<T>& operator*=(const T scalar) { return scalar_mul_inplace(scalar); }

  Matrix transpose() const {
   Matrix<T> result(rows_, columns_);

   for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < columns_; j++) {
     result.at(j,i) = at(i,j);
    }
   }

   return result;
  }

  // Utils
  void print() const {
   for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < columns_; j++) {
     std::cout << std::fixed << std::setprecision(2)
      << std::setw(10) << at(i, j);
    }
    std::cout << std::endl;
   }
  }

  void zeros() {
   data_.assign(rows_ * columns_, 0.0);
  }

  void resize(size_t rows, size_t cols) {
   rows_ = rows;
   columns_ = cols;
   data_.resize(rows * cols);
  }
};

#endif
