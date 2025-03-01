#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include "nn/matrix.hpp"

namespace mnist {
 // read MNIST file format (big endian)
 uint32_t read_big_endian_int(std::ifstream& file) {
  /*
   * reads a 32-bit integer from a file in big-endian format
   * first, it reads 4 bytes from the file into a buffer
   * then it combines these bytes to form a 32-bit integer:
   *   - buf[0] contains the most significant byte (shifted left by 24 bits)
   *   - buf[1] is the second byte (shifted left by 16 bits)
   *   - buf[2] is the third byte (shifted left by 8 bits)
   *   - buf[3] is the least significant byte (used as is)
   * third, the bytes are combined using bitwise OR
   */

  uint8_t buf[4];
  file.read(reinterpret_cast<char*>(buf), 4);
  return (uint32_t)buf[0] << 24 | (uint32_t)buf[1] << 16 | (uint32_t)buf[2] << 8 | (uint32_t)buf[3];
 }
 
 std::vector<Matrix<float> > load_images(const std::string& filename, size_t max_images = -1) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
   throw std::runtime_error("cannot open file: " + filename);
  }
  
  /*
   * MNIST image file format has a 16-byte header:
   *   - 4 bytes: magic number (0x803)
   *   - 4 bytes: number of images
   *   - 4 bytes: number of rows (28)
   *   - 4 bytes: number of columns (28)
   *   - followed by rows*cols bytes for each image
   */

  uint32_t magic = read_big_endian_int(file); // magic number (0x803 for images) - first 4 bytes
  if (magic != 0x803) {
   throw std::runtime_error("invalid MNIST image file format");
  }

  uint32_t num_images = read_big_endian_int(file); // second 4 bytes
  uint32_t rows = read_big_endian_int(file); // third 4 bytes
  uint32_t cols = read_big_endian_int(file); // fourth 4 bytes
    
  if (max_images != -1 && max_images < num_images) {
   num_images = max_images;
  }

  std::vector<Matrix<float> > images;
  images.reserve(num_images);

  for (uint32_t i = 0; i < num_images; ++i) {
   std::vector<float> pixels(rows * cols);
        
   for (uint32_t j = 0; j < rows * cols; ++j) {
    uint8_t pixel;
    file.read(reinterpret_cast<char*>(&pixel), 1);
    pixels[j] = static_cast<float>(pixel) / 255.0f; // normalize pixel values to [0, 1]
   }
        
   // Flatten image into vector
   Matrix<float> image(rows * cols, 1, pixels);
   images.push_back(image);
  }

  return images;
 }

 std::vector<Matrix<float> > load_labels(const std::string& filename, size_t max_labels = -1) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
   throw std::runtime_error("cannot open file: " + filename);
  }

  /*
   * MNIST label file format has an 8-byte header:
   *   - 4 bytes: magic number (0x801)
   *   - 4 bytes: number of labels
   *   - followed by 1 byte for each label
   */

  uint32_t magic = read_big_endian_int(file);
  if (magic != 0x801) { // magic number (0x801 for labels) - first 4 bytes
   throw std::runtime_error("invalid MNIST label file format");
  }

  uint32_t num_labels = read_big_endian_int(file); // second 4 bytes
    
  if (max_labels != -1 && max_labels < num_labels) {
   num_labels = max_labels;
  }

  std::vector<Matrix<float> > labels;
  labels.reserve(num_labels);
 
  for (uint32_t i = 0; i < num_labels; ++i) {
   uint8_t label;
   file.read(reinterpret_cast<char*>(&label), 1);
        
   // Convert to one-hot encoding
   std::vector<float> one_hot(10, 0.0f);
   one_hot[label] = 1.0f;
        
   Matrix<float> label_matrix(10, 1, one_hot);
   labels.push_back(label_matrix);
  }
 
  return labels;
 }

 void visualize_prediction(const Matrix<float>& image, const Matrix<float>& prediction, const Matrix<float>& target) {
  // Determine the actual digit (from target one-hot)
  int actual_digit = 0;
  float max_val = target.at(0, 0);
  for (int i = 1; i < 10; ++i) {
   if (target.at(i, 0) > max_val) {
    max_val = target.at(i, 0);
    actual_digit = i;
   }
  }
    
  // Determine the predicted digit
  int predicted_digit = 0;
  max_val = prediction.at(0, 0);
  for (int i = 1; i < 10; ++i) {
   if (prediction.at(i, 0) > max_val) {
    max_val = prediction.at(i, 0);
    predicted_digit = i;
   }
  }
    
  std::cout << "-------------------------" << std::endl;
  std::cout << "Actual: " << actual_digit << ", Predicted: " << predicted_digit << std::endl;
    
  // Confidence for each digit
  std::cout << "Confidence:" << std::endl;
  for (int i = 0; i < 10; ++i) {
   std::cout << i << ": " << std::fixed << std::setprecision(4) << prediction.at(i, 0) * 100 << "%" << std::endl;
  }
    
  // Print the image as ASCII art
  std::cout << "Image:" << std::endl;
  for (int i = 0; i < 28; ++i) {
   for (int j = 0; j < 28; ++j) {
    float pixel = image.at(i * 28 + j, 0);
    if (pixel < 0.1f) std::cout << " ";
    else if (pixel < 0.3f) std::cout << ".";
    else if (pixel < 0.5f) std::cout << "-";
    else if (pixel < 0.7f) std::cout << "+";
    else if (pixel < 0.9f) std::cout << "*";
    else std::cout << "#";
   }
   std::cout << std::endl;
  }
  std::cout << "-------------------------" << std::endl;
 } 
}
