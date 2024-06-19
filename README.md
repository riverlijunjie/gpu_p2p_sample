# gpu_p2p_sample

Requirement:

    OneAPI: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

Build:

    mkdir build && cd build
    cmake ..
    make


Test:

$ ./sycl_ocl 
Test SYCL OCL backend can access original OCL's cl_mem buffer in the same device:
Device: Intel(R) FPGA Emulation Device, device_type: 8
Device: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz, device_type: 2
Device: Intel(R) Arc(TM) A770 Graphics, device_type: 4
Device: Intel(R) Arc(TM) A770 Graphics, device_type: 4
Chosen device: Intel(R) Arc(TM) A770 Graphics
OCL_SYCL_SHARED: Matrix addition completed successfully.

Test whether L0 backend can access original OpenCL's cl_mem in the same device:
Choose Level-Zero device: Intel(R) Arc(TM) A770 Graphics
Device: Intel(R) FPGA Emulation Device, device_type: 8
Device: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz, device_type: 2
Device: Intel(R) Arc(TM) A770 Graphics, device_type: 4
Device: Intel(R) Arc(TM) A770 Graphics, device_type: 4
Chosen device: Intel(R) Arc(TM) A770 Graphics
Error OCL->USM buffer: Incorrect result at position (0, 0) = 2, ocl_buf = 3
L0 cannot access OCL's USM!

Test whether L0 backend's can access OpenCL backend's USM in the same device:
Choose OpenCL device: Intel(R) Arc(TM) A770 Graphics
Choose Level-Zero device: Intel(R) Arc(TM) A770 Graphics
SYCL L0 Error: Incorrect result at position (0, 0) = 4
L0 cannot access OCL backend's USM!

Test whether SYCL support peer2peer communication for Level-Zero backend:
Choose Level-Zero device: Intel(R) Arc(TM) A770 Graphics
Choose Level-Zero device: Intel(R) Arc(TM) A770 Graphics
SYCL_L0_P2P_communication completed successfully.

Test whether SYCL support peer2peer communication for OpenCL backend:
Choose OpenCL device: Intel(R) Arc(TM) A770 Graphics
Choose OpenCL device: Intel(R) Arc(TM) A770 Graphics
Abort was called at 302 line in file:
./opencl/source/command_queue/enqueue_svm.h
Aborted (core dumped)


