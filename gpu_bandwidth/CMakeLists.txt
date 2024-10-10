add_executable(gpu_bandwidth_test gpu_bandwidth_test.cpp)

target_compile_options(gpu_bandwidth_test PRIVATE -ggdb -O0)
target_link_libraries(gpu_bandwidth_test OpenCL)
