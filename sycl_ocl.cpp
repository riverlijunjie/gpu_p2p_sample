#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <CL/cl.h>
//#include <sycl/sycl.hpp>
//#include <sycl/interop_handle.hpp>

cl_device_id choose_ocl_device()
{
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    for (auto platform : platforms)
    {
        cl_uint deviceCount;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
        for (auto device : devices)
        {
            char deviceName[128];
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            std::cout << "Device: " << deviceName;// << std::endl;
            cl_device_type device_type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
            std::cout << ", device_type: " << device_type << std::endl;
        }

        for (auto device : devices)
        {
            cl_device_type device_type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
            if (device_type == CL_DEVICE_TYPE_GPU)
            {
                char deviceName[128];
                clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
                std::cout << "Chosen device: " << deviceName << std::endl;
                return device;
            }
        }
    }
    return 0;
}
int run_ocl_test() {
    std::cout << "Test simple OpenCL workflow:" << std::endl;
    // Define the size of the matrices
    const int size = 16;
    std::vector<float> A(size * size, 1.0f); // Initialize matrix A with 1.0f
    std::vector<float> B(size * size, 2.0f); // Initialize matrix B with 2.0f
    std::vector<float> C(size * size);       // Matrix C for the result
    // Use the first platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    // Use the first GPU device
    cl_device_id device;
    // clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    device = choose_ocl_device();
    {
        char deviceName[128];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        std::cout << "Chosen Device: " << deviceName; // << std::endl;
        cl_device_type device_type;
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
        std::cout << "device_type: " << device_type << std::endl;
    }
    // Create a context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    // Create a command queue
    // cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, nullptr);
    // Create memory buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, A.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, B.data(), nullptr);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, nullptr, nullptr);
    // Define the kernel source code
    const char* kernelSource = R"(
        __kernel void matrix_add(__global const float* A, __global const float* B, __global float* C) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            C[index] = A[index] + B[index];
            // printf("%f + %f = %f\n",A[index],B[index],C[index]);
        }
    )";
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "matrix_add", nullptr);
    // Set the kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    // Execute the kernel
    size_t globalSize[2] = {size, size};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    // Read the output buffer back to the host
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * size * size, C.data(), 0, nullptr, nullptr);
    // Check the result
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (C[i * size + j] != 3.0f) {
                std::cerr << "Error: Incorrect result at position (" << i << ", " << j << ")" << " = " << C[i * size + j] << "\n";
                return -1;
            }
        }
    }
    std::cout << "OCL Test: Matrix addition completed successfully.\n\n";
    // Clean up
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}

sycl::device choose_sycl_device(bool verbose = false)
{
    std::vector<sycl::device> devices;
    static auto s_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto &d : s_devices)
    {
        if (verbose)
        {
            std::cout << "Platform: " << d.get_platform().get_info<sycl::info::platform::name>();
            std::cout << " -- device name = " << d.get_info<sycl::info::device::name>();
            std::cout << ", backend = " << d.get_backend() << "\n";
        }
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
            devices.emplace_back(d);
    }
    if (devices.size() > 0)
        return devices[0];
    else
        std::cout << "Cannot find correct sycl device!\n";

    return s_devices[0];
}

int run_sycl_test() {
    std::cout << "Test simple SYCL workflow:" << std::endl;
    // Define the size of the matrices
    const int size = 16;
    std::vector<float> A(size * size), B(size * size), C(size * size);
    // Initialize matrices A and B
    for (int i = 0; i < size * size; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    auto device = choose_sycl_device(true);
    // Create a SYCL queue with the OpenCL backend
    sycl::queue queue(device, sycl::property::queue::enable_profiling{});
    // Create buffers for matrices A, B, and C
    sycl::buffer<float, 2> bufA(A.data(), sycl::range<2>(size, size));
    sycl::buffer<float, 2> bufB(B.data(), sycl::range<2>(size, size));
    sycl::buffer<float, 2> bufC(C.data(), sycl::range<2>(size, size));
    // Submit a command group to the queue
    queue.submit([&](sycl::handler& cgh) {
        // Accessors for the buffers
        auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
        auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
        auto accC = bufC.get_access<sycl::access::mode::write>(cgh);
        // Define the kernel
        cgh.parallel_for(sycl::range<2>(size, size), [=](sycl::id<2> idx) {
            int i = idx[0];
            int j = idx[1];
            accC[i][j] = accA[i][j] + accB[i][j];
        });
    });
    // Wait for the queue to finish
    queue.wait();
    // Check the result
    auto accC = bufC.get_access<sycl::access::mode::read>();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (accC[i][j] != 3.0f) {
                std::cerr << "Error: Incorrect result at position (" << i << ", " << j << ")\n";
                return -1;
            }
        }
    }
    std::cout << "SYCL Test: Matrix addition completed successfully.\n\n";
    return 0;
}

int run_ocl_sycl_ocl_test() {
    std::cout << "Test simple OCL->SYCL->OCL workflow:" << std::endl;
    // Define the size of the matrices
    const int size = 16;
    std::vector<float> A(size * size, 1.0f); // Initialize matrix A
    std::vector<float> B(size * size, 2.0f); // Initialize matrix B
    std::vector<float> C(size * size, 3.0f); // Initialize matrix C
    std::vector<float> D(size * size, 4.0f); // Initialize matrix D
    std::vector<float> intermediate(size * size); // Intermediate result for A + B
    std::vector<float> result(size * size);       // Final result for (A + B + C) + D

    // OpenCL context for A + B and final addition with D
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    device = choose_ocl_device();
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, nullptr);

    // Create memory buffers for OpenCL
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, A.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, B.data(), nullptr);
    cl_mem bufIntermediate = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size * size, nullptr, nullptr);

    cl_mem bufD = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, D.data(), nullptr);
    cl_mem bufResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, nullptr, nullptr);
    // Define kernel for matrix addition
    const char* kernelSource = R"(
        __kernel void matrix_add(__global const float* X, __global const float* Y, __global float* Z) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            Z[index] = X[index] + Y[index];
            // printf("%f + %f = %f\n", X[index],Y[index],Z[index]);
        }
    )";
    // Create and build the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_add", nullptr);
    // Set arguments and execute OpenCL kernel for A + B
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufIntermediate);
    size_t globalSize[2] = {size, size};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, bufIntermediate, CL_TRUE, 0, sizeof(float) * size * size, intermediate.data(), 0, nullptr, nullptr);
    
    // SYCL part for intermediate + C
    {
        auto sycl_context = sycl::make_context<::sycl::backend::opencl>(context);
        auto sycl_queue = sycl::make_queue<::sycl::backend::opencl>(queue, sycl_context);
        
        sycl::buffer<float, 1> bufIntermediateSycl(intermediate.data(), sycl::range<1>(size * size));
        sycl::buffer<float, 1> bufC(C.data(), sycl::range<1>(size * size));
        sycl::buffer<float, 1> bufResultSycl(result.data(), sycl::range<1>(size * size));
        sycl_queue.submit([&](sycl::handler& cgh) {
            auto accIntermediate = bufIntermediateSycl.get_access<sycl::access::mode::read>(cgh);
            auto accC = bufC.get_access<sycl::access::mode::read>(cgh);
            auto accResult = bufResultSycl.get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::range<1>(size * size), [=](sycl::id<1> idx) {
                accResult[idx] = accIntermediate[idx] + accC[idx];
            });
        });
        sycl_queue.wait();
    }

    //for (int i = 0; i < size; i++) {
    //    for (int j = 0; j < size; j++) {
    //        std::cout << "After sycl: (" << i << ", " << j << ") = " << result[i * size + j] << "\n";
    //    }
    //}
    // Read back the result from SYCL to host
    {
        // sycl::buffer<float, 1> bufResultSycl(result.data(), sycl::range<1>(size * size));
        // auto accResult = bufResultSycl.get_access<sycl::access::mode::read>();
        // std::copy(accResult.begin(), accResult.end(), result.begin());
    }
    // Final addition with D using OpenCL
    cl_mem bufIntermedia = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, result.data(), nullptr);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufIntermedia);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufD);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufResult);
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, sizeof(float) * size * size, result.data(), 0, nullptr, nullptr);
    // Check the result
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (result[i * size + j] != 10.0f) { // Should be 1 + 2 + 3 + 4 = 10
                std::cerr << "Error: Incorrect result at position (" << i << ", " << j << ")\n";
                return -1;
            }
        }
    }
    std::cout << "OCL_SYCL_OCL: Matrix addition completed successfully.\n";
    // Clean up
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufIntermediate);
    clReleaseMemObject(bufD);
    clReleaseMemObject(bufResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}

#if 1
int run_ocl_sycl_ocl_mem_shared_test() {
    std::cout << "Test OCL->SYCL->OCL without mem shared in same device:" << std::endl;
    // Define the size of the matrices
    const int size = 16;
    std::vector<float> A(size * size, 1.0f); // Initialize matrix A
    std::vector<float> B(size * size, 2.0f); // Initialize matrix B
    std::vector<float> C(size * size, 3.0f); // Initialize matrix C
    std::vector<float> D(size * size, 4.0f); // Initialize matrix D
    //std::vector<float> intermediate(size * size); // Intermediate result for A + B
    std::vector<float> result(size * size);       // Final result for (A + B + C) + D

    // OpenCL context for A + B and final addition with D
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    device = choose_ocl_device();
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, nullptr);

    // Create memory buffers for OpenCL
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, A.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, B.data(), nullptr);

    // Define kernel for matrix addition
    const char* kernelSource = R"(
        __kernel void matrix_add(__global const float* X, __global const float* Y, __global float* Z) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            Z[index] = X[index] + Y[index];
            // printf("%f + %f = %f\n", X[index],Y[index],Z[index]);
        }
    )";
    // Create and build the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_add", nullptr);
    // Set arguments and execute OpenCL kernel for A + B -> B
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufB);
    size_t globalSize[2] = {size, size};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
     
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, C.data(), nullptr);
    // SYCL part for (AB) + C
    {
        auto sycl_context = sycl::make_context<sycl::backend::opencl>(context);
        auto sycl_queue = sycl::make_queue<sycl::backend::opencl>(queue, sycl_context);

#if 0
        sycl::buffer<float, 1> sycl_bufB(bufB, sycl_queue, sycl::range<1>(size * size));
        sycl::buffer<float, 1> sycl_bufC(bufC, sycl_queue, sycl::range<1>(size * size));
#else
        sycl::buffer<float, 1> sycl_bufB(B.data(), sycl::range<1>(size * size));
        sycl::buffer<float, 1> sycl_bufC(C.data(), sycl::range<1>(size * size));
#endif

        sycl_queue.submit([&](sycl::handler& cgh) {
            auto accB = sycl_bufB.get_access<sycl::access::mode::read>(cgh);
            auto accC = sycl_bufC.get_access<sycl::access::mode::read>(cgh);
            auto accResult = sycl_bufC.get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::range<1>(size * size), [=](sycl::id<1> idx) {
                accResult[idx] = accB[idx] + accC[idx];
            });
        });
        sycl_queue.wait();
        auto sycl_res = sycl_bufC.get_access<sycl::access::mode::read>();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                std::cout << "After sycl: (" << i << ", " << j << ") = " << sycl_res[i * size + j] << "\n";
            }
        }
    }

    // Final addition with D using OpenCL - (ABC) + D
    cl_mem bufD = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, D.data(), nullptr);
    cl_mem bufResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufD);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufResult);
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, sizeof(float) * size * size, result.data(), 0, nullptr, nullptr);
    // Check the result
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (result[i * size + j] != 10.0f) { // Should be 1 + 2 + 3 + 4 = 10
                std::cerr << "Error: Incorrect result at position (" << i << ", " << j << ")\n";
                return -1;
            }
        }
    }
    std::cout << "OCL_SYCL_OCL_SHARED: Matrix addition completed successfully.\n";
    // Clean up
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufD);
    clReleaseMemObject(bufResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
#endif

int run_ocl_sycl_ocl_mem_sycl_shared_test() {
    std::cout << "Test OCL->SYCL->OCL with sycl usm shared in same device:" << std::endl;
    // Define the size of the matrices
    const int size = 8;
    std::vector<float> A(size * size, 1.0f); // Initialize matrix A
    std::vector<float> B(size * size, 2.0f); // Initialize matrix B
    std::vector<float> C(size * size, 3.0f); // Initialize matrix C
    std::vector<float> D(size * size, 4.0f); // Initialize matrix D
    std::vector<float> result(size * size);       // Final result for (A + B + C) + D

    // OpenCL context for A + B and final addition with D
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    device = choose_ocl_device();
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, nullptr);

    sycl::context sycl_context = sycl::make_context<sycl::backend::opencl>(context);
    sycl::queue sycl_queue = sycl::make_queue<sycl::backend::opencl>(queue, sycl_context);

    float *bufA = sycl::malloc_shared<float>(size*size, sycl_queue, {});
    float *bufB = sycl::malloc_shared<float>(size*size, sycl_queue, {});
    sycl_queue.memcpy(bufA, A.data(), sizeof(float) * size * size);
    sycl_queue.memcpy(bufB, B.data(), sizeof(float) * size * size);

    // Define kernel for matrix addition
    const char *kernelSource = R"(
        __kernel void matrix_add(__global const float* X, __global const float* Y, __global float* Z) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            printf("%f + %f = %f\n", X[index],Y[index],X[index]+Y[index]);
            Z[index] = X[index] + Y[index];
        }
    )";
    // Create and build the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_add", nullptr);

    // Set arguments and execute OpenCL kernel for A + B -> B
#if 1
    cl_mem cl_bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size * size, bufA, nullptr);
    cl_mem cl_bufB = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size * size, bufB, nullptr);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_bufB);
#else
    //auto cl_bufAA = sycl::get_native<sycl::backend::opencl>(bufA);
    //auto cl_bufBB = sycl::get_native<sycl::backend::opencl>(bufB);

    cl_mem cl_bufA, cl_bufB;
    sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.host_task([=, &cl_bufA](sycl::interop_handle ih) {
            cl_bufA = ih.get_mem_object<sycl::backend::opencl>(bufA);
        });
    }).wait();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_bufA);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_bufA);
#endif
    size_t globalSize[2] = {size, size};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clFlush(queue);
    clFinish(queue);

    float *bufC = sycl::malloc_shared<float>(size*size, sycl_queue, {});
    sycl_queue.memcpy(bufC, C.data(), sizeof(float) * size * size);
    // SYCL part for (AB) + C
    {
        sycl_queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(size * size), [=](sycl::id<1> idx) {
                bufC[idx] = bufB[idx] + bufC[idx];
            });
        });
        sycl_queue.wait();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                std::cout << "After sycl: (" << i << ", " << j << ") = " << bufC[i * size + j] << "\n";
            }
        }
    }

    // Final addition with D using OpenCL - (ABC) + D
    cl_mem cl_bufC = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size * size, bufC, nullptr);
    cl_mem cl_bufD = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, D.data(), nullptr);
    cl_mem bufResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_bufC);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_bufD);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufResult);
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, sizeof(float) * size * size, result.data(), 0, nullptr, nullptr);
    // Check the result
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (result[i * size + j] != 10.0f) { // Should be 1 + 2 + 3 + 4 = 10
                std::cerr << "Error: Incorrect result at position (" << i << ", " << j << ")\n";
                return -1;
            }
        }
    }
    std::cout << "OCL_SYCL_OCL_SHARED: Matrix addition completed successfully.\n";
    // Clean up

    //clReleaseMemObject(cl_bufA);
    //clReleaseMemObject(cl_bufB);
    clReleaseMemObject(cl_bufC);
    clReleaseMemObject(cl_bufD);
    sycl::free(bufA, sycl_queue);
    sycl::free(bufB, sycl_queue);
    sycl::free(bufC, sycl_queue);
    clReleaseMemObject(bufResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}

// Test SYCL OCL backend can access original OCL's cl_mem buffer
int ocl_sycl_test() {
    std::cout << "Test SYCL OCL backend can access original OCL's cl_mem buffer in the same device:\n";
    // Define the size of the matrices
    const int N = 8;
    int matrix_size = N * N;
    // Create a SYCL queue on the default device
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    device = choose_ocl_device();
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, nullptr);

    // Create sycl context based on OCL
    sycl::context sycl_context = sycl::make_context<sycl::backend::opencl>(context);
    sycl::queue sycl_queue = sycl::make_queue<sycl::backend::opencl>(queue, sycl_context);

    // Allocate memory for matrices A, B, and C
    float* A = sycl::malloc_shared<float>(matrix_size, sycl_queue, {});
    float* B = sycl::malloc_shared<float>(matrix_size, sycl_queue,{});
    float* C = sycl::malloc_shared<float>(matrix_size, sycl_queue,{});
    // Initialize matrices A, B, and C on host side, not trigger to update into device side
    for (int i = 0; i < matrix_size; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
    }
    std::vector<float> AA(N * N, 1.0f); // Initialize matrix A
    std::vector<float> BB(N * N, 2.0f); // Initialize matrix B
    // Must copy data to device side, because clCreateBuffer will not trigger copy data to device side
    sycl_queue.memcpy(A, AA.data(), sizeof(float) * N * N);
    sycl_queue.memcpy(B, BB.data(), sizeof(float) * N * N);

    // Map data to cl_mem without data copying? Need confirm.
    cl_mem cl_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * matrix_size, A, nullptr);
    cl_mem cl_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * matrix_size, B, nullptr);

    // OpenCL operations to add matrix A to matrix B
    const char *kernel_source = R"(
        __kernel void matrix_add(__global const float* A, __global float* B) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            B[index] += A[index];
            // printf("A = %f, B = %f\n", A[index], B[index]);
        }
    )";
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_add", nullptr);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_B);
    size_t global_work_size[2] = {N, N};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    clFlush(queue);
    clFinish(queue);
    clReleaseMemObject(cl_A);
    clReleaseMemObject(cl_B);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    // cl_B store the A+B result, it will map to sycl usm B
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (B[i * N + j] != 3.0f)
            { // Should be 1 + 2 = 3
                std::cerr << "Error OCL->USM buffer: Incorrect result at position (" << i << ", " << j << ") = " << B[i * N + j] << "\n\n";
                return -1;
            }
        }
    }

    // Use SYCL to add matrix B to matrix C
    sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(matrix_size), [=](sycl::id<1> idx) {
            C[idx] += B[idx];
        });
    }).wait();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i * N + j] != 6.0f) { // Should be 1 + 2 + 3 = 6
                std::cerr << "Error SYCL result: Incorrect result at position (" << i << ", " << j << ") = " << C[i * N + j] << "\n\n";
                return -1;
            }
        }
    }
    std::cout << "OCL_SYCL_SHARED: Matrix addition completed successfully.\n\n";

    // Free the memory
    sycl::free(A, sycl_queue);
    sycl::free(B, sycl_queue);
    sycl::free(C, sycl_queue);
    return 0;
}

// device_type = 0: L0 device
// device_type = 1: OpenCL device
// other: L0 device
sycl::device get_sycl_device(int device_type = 0, int index = 0, bool verbose = false)
{
    std::vector<sycl::device> ocl_devices;
    std::vector<sycl::device> l0_devices;
    if (verbose)
        std::cout << "Platform list:" << std::endl;
    for (auto &p : sycl::platform::get_platforms())
    {
        std::string platform_name = p.get_info<sycl::info::platform::name>();
        if (verbose)
            std::cout << "\t" << platform_name << "\n";
        auto devices = p.get_devices(sycl::info::device_type::all);
        for (auto &d : devices)
        {
            if (platform_name.find("OpenCL Graphics") != std::string::npos)
                ocl_devices.emplace_back(d);
            if (platform_name.find("Level-Zero") != std::string::npos)
                l0_devices.emplace_back(d);
            if (verbose)
                std::cout << "\t\t" << d.get_info<sycl::info::device::name>() << "\n";
        }
    }

    sycl::platform p;
    auto devices = p.get_devices(sycl::info::device_type::gpu);
    if (verbose)
    {
        std::cout << "Choose Device list of " << p.get_info<sycl::info::platform::name>() << ":" << std::endl;
        for (auto &d : devices)
        {
            std::cout << "\t" << d.get_info<sycl::info::device::name>() << "\n";
        }
    }
    if (device_type == 1)
    {
        std::cout << "Choose OpenCL device: " << ocl_devices[index].get_info<sycl::info::device::name>() << "\n";
        return ocl_devices[index];
    }

    std::cout << "Choose Level-Zero device: " << l0_devices[index].get_info<sycl::info::device::name>() << "\n";
    return l0_devices[index];
}

// Test whether L0 backend can access original OpenCL's cl_mem in the same device
int ocl_sycl_L0_test() {
    std::cout << "Test whether L0 backend can access original OpenCL's cl_mem in the same device:\n";
    // Define the size of the matrices
    const int N = 8;
    int matrix_size = N * N;
    // zeInit(0);
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // Get device with Level-Zero
    auto sycl_device = get_sycl_device(0,0);
    sycl::queue sycl_queue(sycl_device,
                           sycl::property_list{
                               sycl::property::queue::enable_profiling(),
                               sycl::property::queue::in_order()});
    auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_queue.get_device());
    auto l0_ctx = sycl::get_native<
        sycl::backend::ext_oneapi_level_zero>(sycl_queue.get_context());

    // Choose device with OpenCL graphics
    cl_device_id device = choose_ocl_device();
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, nullptr);
    // Cannot creat Level-Zero context with OCL context
    //sycl::context sycl_context = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(context);
    //sycl::queue sycl_queue = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(queue, sycl_context);

    // Allocate memory for matrices A, B, and C
    float* A = sycl::malloc_shared<float>(matrix_size, sycl_queue,{});
    float* B = sycl::malloc_shared<float>(matrix_size, sycl_queue,{});
    float* C = sycl::malloc_shared<float>(matrix_size, sycl_queue,{});
    // Initialize matrices A, B, and C
    for (int i = 0; i < matrix_size; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
    }
    std::vector<float> AA(N * N, 1.0f); // Initialize matrix A
    std::vector<float> BB(N * N, 2.0f); // Initialize matrix B
    sycl_queue.memcpy(A, AA.data(), sizeof(float) * N * N);
    sycl_queue.memcpy(B, BB.data(), sizeof(float) * N * N);

    // OpenCL operations to add matrix A to matrix B
    cl_mem cl_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * matrix_size, A, nullptr);
    cl_mem cl_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * matrix_size, B, nullptr);

    const char *kernel_source = R"(
        __kernel void matrix_add(__global const float* A, __global float* B) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            B[index] += A[index];
            // printf("A = %f, B = %f\n", A[index], B[index]);
        }
    )";
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_add", nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_B);
    size_t global_work_size[2] = {N, N};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    clFlush(queue);
    clFinish(queue);

    // Read the output buffer back to the host
    std::vector<float> cl_result(N * N);
    clEnqueueReadBuffer(queue, cl_B, CL_TRUE, 0, sizeof(float) * N * N, cl_result.data(), 0, nullptr, nullptr);

    clReleaseMemObject(cl_A);
    clReleaseMemObject(cl_B);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    // cl_B store the A+B result, it will map to sycl usm B
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if ((B[i * N + j] != 3.0f) || (cl_result[i * N + j] != 3.0f))
            { // Should be 1 + 2 = 3
                std::cerr << "Error OCL->USM buffer: Incorrect result at position (" << i << ", " << j << ") = " << B[i * N + j] << ", ocl_buf = " << cl_result[i * N + j] << "\n";
                std::cerr << "L0 cannot access OCL's USM!\n\n";
                return -1;
            }
        }
    }

    // Use SYCL to add matrix B to matrix C
    sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(matrix_size), [=](sycl::id<1> idx) {
            C[idx] += B[idx];
        });
    }).wait();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i * N + j] != 6.0f) { // Should be 1 + 2 + 3 = 6
                std::cerr << "Error: Incorrect result at position (" << i << ", " << j << ") = " <<C[i * N + j] <<"\n";
                return -1;
            }
        }
    }
    std::cout << "OCL_SYCL_L0_SHARED: Matrix addition completed successfully.\n";

    // Free the memory
    sycl::free(A, sycl_queue);
    sycl::free(B, sycl_queue);
    sycl::free(C, sycl_queue);
    return 0;
}

// Test whether L0 backend's can access OpenCL backend's USM in the same device
int sycl_ocl_and_L0_test() {
    std::cout << "Test whether L0 backend's can access OpenCL backend's USM in the same device:" << std::endl;
    // Define the size of the matrices
    const int N = 8;
    int matrix_size = N * N;
    // zeInit(0);

    // Get OpenCL device
    auto sycl_ocl_device = get_sycl_device(1,0);
    sycl::queue sycl_ocl_queue(sycl_ocl_device,
                           sycl::property_list{
                               sycl::property::queue::enable_profiling(),
                               sycl::property::queue::in_order()});
#if 0
    auto ocl_device = sycl::get_native<sycl::backend::opencl>(sycl_ocl_queue.get_device());
    auto ocl_ctx = sycl::get_native<
        sycl::backend::opencl>(sycl_ocl_queue.get_context());
#endif

    // Get Level-Zero device
    auto sycl_l0_device = get_sycl_device(0,0);
    sycl::queue sycl_l0_queue(sycl_l0_device,
                           sycl::property_list{
                               sycl::property::queue::enable_profiling(),
                               sycl::property::queue::in_order()});
#if 0
    auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_l0_queue.get_device());
    auto l0_ctx = sycl::get_native<
        sycl::backend::ext_oneapi_level_zero>(sycl_l0_queue.get_context());
#endif

    // Allocate memory for matrices A, B, and C
    float* ocl_A = sycl::malloc_shared<float>(matrix_size, sycl_ocl_queue,{});
    float* ocl_B = sycl::malloc_shared<float>(matrix_size, sycl_ocl_queue,{});
    float* l0_C = sycl::malloc_shared<float>(matrix_size, sycl_l0_queue,{});
    // Initialize matrices A, B, and C
    for (int i = 0; i < matrix_size; ++i) {
        ocl_A[i] = 1.0f;
        ocl_B[i] = 2.0f;
        l0_C[i] = 4.0f;
    }
    std::vector<float> AA(N * N, 1.0f); // Initialize matrix A
    std::vector<float> BB(N * N, 2.0f); // Initialize matrix B
    std::vector<float> CC(N * N, 4.0f); // Initialize matrix c
    sycl_ocl_queue.memcpy(ocl_A, AA.data(), sizeof(float) * N * N);
    sycl_ocl_queue.memcpy(ocl_B, BB.data(), sizeof(float) * N * N);
    sycl_l0_queue.memcpy(l0_C, CC.data(), sizeof(float) * N * N);

    // Use OCL SYCL to add matrix A to matrix B
    sycl_ocl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(matrix_size), [=](sycl::id<1> idx) {
            ocl_B[idx] += ocl_A[idx];
        });
    }).wait();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (ocl_B[i * N + j] != 3.0f) { // Should be 1 + 2 = 3
                std::cerr << "SYCL OCL Error: Incorrect result at position (" << i << ", " << j << ") = " <<ocl_B[i * N + j] <<"\n";
                return -1;
            }
        }
    }

    // Use OCL L0 to add matrix B to matrix C
    sycl_l0_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(matrix_size), [=](sycl::id<1> idx) {
            l0_C[idx] += ocl_B[idx];
        });
    }).wait();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (l0_C[i * N + j] != 7.0f) { // Should be 3 + 4 = 7
                std::cerr << "SYCL L0 Error: Incorrect result at position (" << i << ", " << j << ") = " <<l0_C[i * N + j] <<"\n";
                std::cerr << "L0 cannot access OCL backend's USM!\n\n";
                return -1;
            }
        }
    }

    std::cout << "SYCL_OCL_and_L0_SHARED: Matrix addition completed successfully.\n";

    // Free the memory
    sycl::free(ocl_A, sycl_ocl_queue);
    sycl::free(ocl_B, sycl_ocl_queue);
    sycl::free(l0_C, sycl_l0_queue);
    return 0;
}

// Test whether sycl support peer2peer communication
// backend = 1: OpenCL
// backend = 0: L0
int sycl_peer_to_peer_test(int backend)
{
    std::cout << "Test whether SYCL support peer2peer communication for ";
    if(backend==1)
        std::cout << "OpenCL backend:\n";
    else
        std::cout << "Level-Zero backend:\n";
    // Define the size of the matrices
    const int N = 8;
    int matrix_size = N * N;
    // zeInit(0);

    // Get device 0
    auto sycl_device_0 = get_sycl_device(backend, 0);
    sycl::queue sycl_queue_0(sycl_device_0,
                             sycl::property_list{
                                 sycl::property::queue::enable_profiling(),
                                 sycl::property::queue::in_order()});

    // Get device 1
    auto sycl_device_1 = get_sycl_device(backend, 1);
    sycl::queue sycl_queue_1(sycl_device_1,
                             sycl::property_list{
                                 sycl::property::queue::enable_profiling(),
                                 sycl::property::queue::in_order()});

    // Allocate memory for usm 0 in device 0, and usm 1 in device 1
    float *usm_0 = sycl::malloc_shared<float>(matrix_size, sycl_queue_0, {});
    float *usm_1 = sycl::malloc_shared<float>(matrix_size, sycl_queue_1, {});
    float *usm_2 = sycl::malloc_shared<float>(matrix_size, sycl_queue_1, {});

    std::vector<float> A(N * N, 1.0f);  // Initialize 1.0
    std::vector<float> B(N * N, 10.0f); // Initialize 10.0
    // std::vector<float> C(N * N, 100.0f); // Initialize 100.0
    sycl_queue_0.memcpy(usm_0, A.data(), sizeof(float) * N * N);
    sycl_queue_1.memcpy(usm_1, B.data(), sizeof(float) * N * N);
    // sycl_queue_1.memcpy(usm_2, C.data(), sizeof(float) * N * N);

    // usm_0 *= 3
    sycl_queue_0.submit([&](sycl::handler &cgh)
                        { cgh.parallel_for(sycl::range<1>(matrix_size), [=](sycl::id<1> idx)
                                           { usm_0[idx] *= 3; }); })
        .wait();

    // copy usm_0 to usm_1
    sycl_queue_1.memcpy(usm_2, usm_0, sizeof(float) * N * N);

    // usm_2 = usm_0 * 3 + usm_1 = 13
    sycl_queue_1.submit([&](sycl::handler &cgh)
                        { cgh.parallel_for(sycl::range<1>(matrix_size), [=](sycl::id<1> idx)
                                           { usm_2[idx] += usm_1[idx]; }); })
        .wait();

    for (int i = 0; i < N * N; i++)
    {
        if (usm_2[i] != 13.0f)
        { // Should be 1*3 + 10 = 13
            if (backend == 0)
                std::cerr << "SYCL L0 ";
            else
                std::cerr << "SYCL OCL ";
            std::cerr << " Error: Incorrect result at position (" << i << ") = " << usm_2[i] << "\n\n";
            return -1;
        }
    }

    if (backend == 0)
        std::cout << "SYCL_L0_P2P_communication completed successfully.\n\n";
    else
        std::cout << "SYCL_OCL_P2P_communication completed successfully.\n\n";

    // Free the memory
    sycl::free(usm_0, sycl_queue_0);
    sycl::free(usm_1, sycl_queue_1);
    sycl::free(usm_2, sycl_queue_1);
    return 0;
}

int main() {
    //get_sycl_device(0,0,true);
    //run_ocl_test();
    //run_sycl_test();
    //run_ocl_sycl_ocl_test();

    //run_ocl_sycl_ocl_mem_shared_test();
    // run_ocl_sycl_ocl_mem_sycl_shared_test();
    ocl_sycl_test();
    ocl_sycl_L0_test();
    sycl_ocl_and_L0_test();
    sycl_peer_to_peer_test(0);
    sycl_peer_to_peer_test(1);
    return 0;
}