add_executable(ocl_matmul_test gpu_matmul.cpp)

#target_compile_options(ocl_matmul_test PRIVATE -ggdb -O0)
target_compile_options(ocl_matmul_test PRIVATE -O3)
target_link_libraries(ocl_matmul_test OpenCL)
