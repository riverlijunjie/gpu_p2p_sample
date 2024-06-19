cmake_minimum_required(VERSION 3.10)
# Set the project name and language
project(SYCL_OCL_Project LANGUAGES CXX)

# Set the C++ compiler to icpx
set(CMAKE_CXX_COMPILER "icpx")

#include_directories("/opt/intel/oneapi/compiler/2024.1/include/sycl"
#                    "/opt/intel/oneapi/compiler/2024.1/include")

# Add executable target with source file
add_executable(sycl_ocl sycl_ocl.cpp)

# Add compile options
target_compile_options(sycl_ocl PRIVATE -ggdb -O0 -fsycl)

# Link against the OpenCL library
target_link_libraries(sycl_ocl PRIVATE OpenCL -fsycl)

