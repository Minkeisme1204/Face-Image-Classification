cmake_minimum_required(VERSION 3.0)

project(CYOLO VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/configs)
# add src dir
# add_subdirectory(src)

# Define and Add Custom Libraries
add_library(numpy STATIC src/numpy.c)

add_library(activation STATIC src/activation.c)
target_link_libraries(activation numpy m)

add_library(convolution STATIC src/convolution.c)
target_link_libraries(convolution activation)

# Testing Programs Declararations 
enable_testing()
add_executable(test_numpy test/numpy_test.c)
target_link_libraries(test_numpy numpy)

enable_testing()
add_executable(test_conv test/conv_test.c)
target_link_libraries(test_conv convolution m)

add_test(NAME RunNumpyTest COMMAND test_numpy)
add_test(NAME RunPaddingTest COMMAND test_conv)
# add_test(NAME RunConvolutionTest COMMAND test_conv2)