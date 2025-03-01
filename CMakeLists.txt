cmake_minimum_required(VERSION 3.10)
project(nn_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
     googletest
     URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Create a library for mnist utilities
add_library(mnist_utils src/mnist_utils.cpp)

# Add examples
add_executable(mnist src/mnist.cpp)

# Link libraries
target_link_libraries(mnist PRIVATE mnist_utils)

# Add tests directory
add_subdirectory(tests)

install(TARGETS mnist
        RUNTIME DESTINATION bin)

install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "Project configured successfully")
