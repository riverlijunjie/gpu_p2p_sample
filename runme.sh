source /opt/intel/oneapi/compiler/latest/env/vars.sh
icpx -o sycl_ocl sycl_ocl.cpp -fsycl -ggdb -O0 -lOpenCL
