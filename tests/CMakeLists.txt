# Create test executable
add_executable(matrix_tests matrix_tests.cpp)

# Link against GTest
target_link_libraries(matrix_tests PRIVATE GTest::gtest_main)

# Enable testing
include(GoogleTest)
gtest_discover_tests(matrix_tests)
