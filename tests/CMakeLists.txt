# Create test executable
add_executable(matrix_tests matrix_tests.cpp)
add_executable(activation_tests activation_tests.cpp)
add_executable(optimizer_tests optimizer_tests.cpp)
add_executable(layer_tests layer_tests.cpp)
add_executable(network_tests network_tests.cpp)

# Link against GTest
target_link_libraries(matrix_tests PRIVATE GTest::gtest_main)
target_link_libraries(activation_tests PRIVATE GTest::gtest_main)
target_link_libraries(optimizer_tests PRIVATE GTest::gtest_main)
target_link_libraries(layer_tests PRIVATE GTest::gtest_main)
target_link_libraries(network_tests PRIVATE GTest::gtest_main)

# Enable testing
include(GoogleTest)
gtest_discover_tests(matrix_tests)
gtest_discover_tests(activation_tests)
gtest_discover_tests(optimizer_tests)
gtest_discover_tests(layer_tests)
gtest_discover_tests(network_tests)
