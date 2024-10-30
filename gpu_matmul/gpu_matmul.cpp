#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <map>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <ctime>

#define DEBUG_ENABLE false
#define LOOP_TIMES 20

#define CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL (1 << 23)
#define CL_MEM_ALLOCATION_HANDLE_INTEL 0x10050
static std::map<int, std::string> oclErrorCode =
    {
        {0, "CL_SUCCESS"},
        {-1, "CL_DEVICE_NOT_FOUND"},
        {-2, "CL_DEVICE_NOT_AVAILABLE"},
        {-3, "CL_COMPILER_NOT_AVAILABLE"},
        {-4, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
        {-5, "CL_OUT_OF_RESOURCES"},
        {-6, "CL_OUT_OF_HOST_MEMORY"},
        {-7, "CL_PROFILING_INFO_NOT_AVAILABLE"},
        {-8, "CL_MEM_COPY_OVERLAP"},
        {-9, "CL_IMAGE_FORMAT_MISMATCH"},
        {-10, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
        {-11, "CL_BUILD_PROGRAM_FAILURE"},
        {-12, "CL_MAP_FAILURE"},
        {-13, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
        {-14, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
        {-15, "CL_COMPILE_PROGRAM_FAILURE"},
        {-16, "CL_LINKER_NOT_AVAILABLE"},
        {-17, "CL_LINK_PROGRAM_FAILURE"},
        {-18, "CL_DEVICE_PARTITION_FAILED"},
        {-19, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
        {-30, "CL_INVALID_VALUE"},
        {-31, "CL_INVALID_DEVICE_TYPE"},
        {-32, "CL_INVALID_PLATFORM"},
        {-33, "CL_INVALID_DEVICE"},
        {-34, "CL_INVALID_CONTEXT"},
        {-35, "CL_INVALID_QUEUE_PROPERTIES"},
        {-36, "CL_INVALID_COMMAND_QUEUE"},
        {-37, "CL_INVALID_HOST_PTR"},
        {-38, "CL_INVALID_MEM_OBJECT"},
        {-39, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
        {-40, "CL_INVALID_IMAGE_SIZE"},
        {-41, "CL_INVALID_SAMPLER"},
        {-42, "CL_INVALID_BINARY"},
        {-43, "CL_INVALID_BUILD_OPTIONS"},
        {-44, "CL_INVALID_PROGRAM"},
        {-45, "CL_INVALID_PROGRAM_EXECUTABLE"},
        {-46, "CL_INVALID_KERNEL_NAME"},
        {-47, "CL_INVALID_KERNEL_DEFINITION"},
        {-48, "CL_INVALID_KERNEL"},
        {-49, "CL_INVALID_ARG_INDEX"},
        {-50, "CL_INVALID_ARG_VALUE"},
        {-51, "CL_INVALID_ARG_SIZE"},
        {-52, "CL_INVALID_KERNEL_ARGS"},
        {-53, "CL_INVALID_WORK_DIMENSION"},
        {-54, "CL_INVALID_WORK_GROUP_SIZE"},
        {-55, "CL_INVALID_WORK_ITEM_SIZE"},
        {-56, "CL_INVALID_GLOBAL_OFFSET"},
        {-57, "CL_INVALID_EVENT_WAIT_LIST"},
        {-58, "CL_INVALID_EVENT"},
        {-59, "CL_INVALID_OPERATION"},
        {-60, "CL_INVALID_GL_OBJECT"},
        {-61, "CL_INVALID_BUFFER_SIZE"},
        {-62, "CL_INVALID_MIP_LEVEL"},
        {-63, "CL_INVALID_GLOBAL_WORK_SIZE"},
        {-64, "CL_INVALID_PROPERTY"},
        {-65, "CL_INVALID_IMAGE_DESCRIPTOR"},
        {-66, "CL_INVALID_COMPILER_OPTIONS"},
        {-67, "CL_INVALID_LINKER_OPTIONS"},
        {-68, "CL_INVALID_DEVICE_PARTITION_COUNT"},
        {-69, "CL_INVALID_PIPE_SIZE"},
        {-70, "CL_INVALID_DEVICE_QUEUE"},
        {-71, "CL_INVALID_SPEC_ID"},
        {-72, "CL_MAX_SIZE_RESTRICTION_EXCEEDED"},
};

#define CHECK_OCL_ERROR(err, msg)                                                                                          \
    if (err < 0)                                                                                                           \
    {                                                                                                                      \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown";               \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n", __FUNCTION__, __LINE__, msg, err, errstr.c_str()); \
    }

#define CHECK_OCL_ERROR_RETURN(err, msg, ret)                                                                              \
    if (err < 0)                                                                                                           \
    {                                                                                                                      \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown";               \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n", __FUNCTION__, __LINE__, msg, err, errstr.c_str()); \
        return ret;                                                                                                        \
    }

#define CHECK_OCL_ERROR_EXIT(err, msg)                                                                                     \
    if (err < 0)                                                                                                           \
    {                                                                                                                      \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown";               \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n", __FUNCTION__, __LINE__, msg, err, errstr.c_str()); \
        exit(1);                                                                                                           \
    }

static int g_device_num = 0;
cl_device_id choose_ocl_device(size_t id = 0, std::vector<std::string> extensions = {})
{
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    std::vector<cl_device_id> devices_vec;
    cl_int err;
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
            // std::cout << "Device: " << deviceName; // << std::endl;
            cl_device_type device_type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
            // std::cout << ", device_type: " << device_type << std::endl;
        }

        for (auto device : devices)
        {
            cl_device_type device_type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
            if (device_type == CL_DEVICE_TYPE_GPU)
            {
                char supported_extensions[4096];
                err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(supported_extensions), &supported_extensions, nullptr);
                CHECK_OCL_ERROR(err, "clGetDeviceInfo(CL_DEVICE_EXTENSIONS)");
                std::string full_extensions(supported_extensions);
                bool supported = true;
                for (auto &ext : extensions)
                {
                    if (full_extensions.find(ext) == std::string::npos)
                    {
                        supported = false;
                        break;
                    }
                }
                if (supported)
                    devices_vec.push_back(device);
            }
        }
    }
    g_device_num = devices_vec.size();
    std::cout << "GPU device number = " << devices_vec.size() << std::endl;
    if (devices_vec.size() > id)
    {
        char deviceName[128];
        clGetDeviceInfo(devices_vec[id], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        std::cout << "Choose device: id = " << id << ", name = " << deviceName << std::endl;
        return devices_vec[id];
    }
    else
    {
        std::cout << "Error: cannot choose device: id = " << id << std::endl;
        exit(2);
    }
    return 0;
}

void init_matrix(std::vector<int8_t> &A, std::vector<int8_t> &B, std::vector<int8_t> &C, size_t M, size_t N, size_t K)
{

    for (size_t j = 0; j < M; j++)
    {
        for (size_t i = 0; i < K; i++)
        {
            A[j * K + i] = j * K + i;//(i * M + j + i) % 7 - 4;
        }
    }
    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < K; i++)
        {
            B[j * K + i] = j * K + i;//(i * N + j + i) % 5 - 2;
        }
    }
    for (size_t j = 0; j < M; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            C[j * N + i] = 0;//(i * K + j + i) % 7;
        }
    }
}

void print_matrix(std::vector<int8_t> &A, std::vector<int8_t> &B, std::vector<int8_t> &C, size_t M, size_t N, size_t K)
{

    std::cout << "Matrix A:" << std::endl;
    for (size_t j = 0; j < M; j++)
    {
        for (size_t i = 0; i < K; i++)
        {
            std::cout << " " << static_cast<int>(A[j * K + i]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
    std::cout << "Matrix B:" << std::endl;
    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < K; i++)
        {
            std::cout << " " << static_cast<int>(B[j * K + i]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl
              << std::endl;
    std::cout << "Matrix C:" << std::endl;
    for (size_t j = 0; j < M; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            std::cout << " " << static_cast<int>(C[j * N + i]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl
              << std::endl;
}

std::vector<int> matmul_cpu(size_t M, size_t N, size_t K)
{
    std::vector<int8_t> A(M * K, 1);
    std::vector<int8_t> B(N * K, 2);
    std::vector<int8_t> C(M * N, 3);
    std::vector<int> D(M * N, 0);

    init_matrix(A, B, C, M, N, K);
    const auto start_1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            int sum = C[i * N + j];
            for (size_t kk = 0; kk < K; kk++)
                sum += A[i * K + kk] * B[j * K + kk];
            D[i * N + j] = sum;
        }
    }
    const auto end_1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_1 = end_1 - start_1;
    std::cout << "matmul_cpu cost: " << elapsed_1.count() << " ms" << std::endl;
    return D;
}

void matmul_gpu_base(cl_command_queue queue, cl_program program, std::string kernel_name,
                     cl_mem bufA, cl_mem bufB, cl_mem bufC, cl_mem bufD,
                     std::vector<int> &reference,
                     size_t M, size_t N, size_t K)
{
    std::vector<int> result(M * N, 0);
    cl_int err;

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateKernel failed");

    for (size_t loop = 0; loop < 1; loop++)
    {
        // Set the kernel arguments
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufD);
        clSetKernelArg(kernel, 4, sizeof(int), &M);
        clSetKernelArg(kernel, 5, sizeof(int), &N);
        clSetKernelArg(kernel, 6, sizeof(int), &K);
        // Execute the kernel
        cl_int err;
        size_t globalSize[2] = {M, N};
        const auto start_1 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < LOOP_TIMES; i++)
            err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue);
        const auto end_1 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_1 = end_1 - start_1;
        std::cout << kernel_name << " cost: " << elapsed_1.count() /LOOP_TIMES << " ms" << std::endl;
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");

        err = clEnqueueReadBuffer(queue, bufD, CL_TRUE, 0, sizeof(float) * M * N, result.data(), 0, nullptr, nullptr);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");

        size_t cnt = 0;
        for (int i = 0; i < M * N; i += 1)
        {
            if (reference[i] != result[i])
            {
                cnt += 1;
            }
        }
        if (cnt == 0)
            std::cout << kernel_name << " check: SUCCESS" << std::endl
                      << std::endl;
        else
            std::cout << kernel_name << " check: FAIL" << "(" << 1.0 * cnt / (M * N) << ")" << std::endl
                      << std::endl;
    }
    clReleaseKernel(kernel);
}

void matmul_gpu_bf_tile(cl_command_queue queue, cl_program program, std::string kernel_name,
                        cl_mem bufA, cl_mem bufB, cl_mem bufC, cl_mem bufD,
                        std::vector<int> &reference,
                        size_t M, size_t N, size_t K, size_t gws[3], size_t lws[3])
{
    std::vector<int> result(M * N, 0);
    cl_int err;

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateKernel failed");

    for (size_t loop = 0; loop < 1; loop++)
    {
        // Set the kernel arguments
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufD);
        clSetKernelArg(kernel, 4, sizeof(int), &M);
        clSetKernelArg(kernel, 5, sizeof(int), &N);
        clSetKernelArg(kernel, 6, sizeof(int), &K);
        // Execute the kernel
        //#define TILE_B 8
        //#define TILE_OFM 1
        //size_t gws[3] = {M * N / TILE_B / TILE_OFM, 1, 1};
        //size_t lws[3] = {16, 1, 1};
        const auto start_1 = std::chrono::high_resolution_clock::now();
        cl_int err;
        for(int i = 0; i < LOOP_TIMES; i++)
            err = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, gws, lws, 0, nullptr, nullptr);
        clFinish(queue);
        const auto end_1 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_1 = end_1 - start_1;
        std::cout << kernel_name << " cost: " << elapsed_1.count() / LOOP_TIMES << " ms" << std::endl;
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");

        err = clEnqueueReadBuffer(queue, bufD, CL_TRUE, 0, sizeof(float) * M * N, result.data(), 0, nullptr, nullptr);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");

        size_t cnt = 0;
        for (int i = 0; i < M * N; i += 1)
        {
            if (reference[i] != result[i])
            {
                cnt += 1;
            }
            if (i < 32 && DEBUG_ENABLE)
            {
                std::cout << "(" << reference[i] << "," << result[i] << ") ";
                if (i % 8 == 7)
                    std::cout << std::endl;
            }
        }
        std::cout << std::endl;
        if (cnt == 0)
            std::cout << kernel_name << " check: SUCCESS" << std::endl
                      << std::endl;
        else
            std::cout << kernel_name << " check: FAIL" << "(" << 1.0 * cnt / (M * N) << ")" << std::endl
                      << std::endl;
    }
    clReleaseKernel(kernel);
}

void matmul_gpu_bf_xmx(cl_command_queue queue, cl_program program, std::string kernel_name,
                       cl_mem bufA, cl_mem bufB, cl_mem bufC, cl_mem bufD,
                       std::vector<int> &reference,
                       size_t M, size_t N, size_t K, size_t gws[3], size_t lws[3])
{
    std::vector<int> result(M * N, 0);
    cl_int err;

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateKernel failed");

    for (size_t loop = 0; loop < 1; loop++)
    {
        // Set the kernel arguments
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufD);
        clSetKernelArg(kernel, 4, sizeof(int), &M);
        clSetKernelArg(kernel, 5, sizeof(int), &N);
        clSetKernelArg(kernel, 6, sizeof(int), &K);
        // Execute the kernel
        // #define TILE_B 8
        // #define TILE_F 8
        cl_int err;
        // size_t gws[3] = {M * N / TILE_B, 1, 1};
        // size_t lws[3] = {8, 1, 1};
        const auto start_1 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < LOOP_TIMES; i++)
            err = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, gws, lws, 0, nullptr, nullptr);
        clFinish(queue);
        const auto end_1 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_1 = end_1 - start_1;
        std::cout << kernel_name << " cost: " << elapsed_1.count() / LOOP_TIMES << " ms" << std::endl;
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");

        err = clEnqueueReadBuffer(queue, bufD, CL_TRUE, 0, sizeof(float) * M * N, result.data(), 0, nullptr, nullptr);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");

        size_t cnt = 0;
        for (int i = 0; i < M; i += 1)
        {
            for (int j = 0; j < N; j++)
            {
                if (reference[i * N + j] != result[i * N + j])
                {
                    cnt += 1;
                }
                if (DEBUG_ENABLE)
                {
                    if (i < 16 && j < 16)
                    {
                        std::cout << "(" << reference[i * N + j] << "," << result[i * N + j] << ") ";
                    }
                    if (j % N == N - 1 && i < 16)
                        std::cout << std::endl;
                }
            }
        }
        std::cout << std::endl;
        if (cnt == 0)
            std::cout << kernel_name << " check: SUCCESS" << std::endl
                      << std::endl;
        else
            std::cout << kernel_name << " check: FAIL" << "(" << 1.0 * cnt / (M * N) << ")" << std::endl
                      << std::endl;
    }
    clReleaseKernel(kernel);
}

void run_matmul_test(cl_device_id device, size_t M, size_t N, size_t K)
{
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, nullptr);

    cl_int err;
    std::vector<int8_t> A(M * K, 1);
    std::vector<int8_t> B(N * K, 2);
    std::vector<int8_t> C(M * N, 3);
    std::vector<int> D(M * N, 0);
    init_matrix(A, B, C, M, N, K);
    std::vector<int> reference = matmul_cpu(M, N, K);

    // D = A * B + C
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t) * M * K, A.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t) * N * K, B.data(), nullptr);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t) * M * N, C.data(), nullptr);
    cl_mem bufD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * M * N, D.data(), nullptr);

    // Define the kernel source code
    const char *kernelSource = R"(
        #include "../matmul_bf_tile.cl"
        #include "../matmul_bf_xmx.cl"
        __kernel void matmul_base(__global const char* A, __global const char* B, __global char* C, __global int* D, int M, int N, int K) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * N + j;
            int temp = C[index];
            unroll_for(uint item = 0; item < K; item++) {
                // printf("\tA[%d]=%d, b[%d]=%d\n",i*K+item, A[i*K+item], j*K+item, B[j*K+item]);
                temp += A[i*K+item] * B[j*K+item];
            }
            D[index] = temp;
            // printf("(%ld,%ld): index = %ld, D[index] = %d\n",i,j,index, D[index]);
        }
    )";
    // Create a program from the kernel source
    cl_uint knlcount = 1;
    const char *knlstrList[] = {kernelSource};
    size_t knlsizeList[] = {strlen(kernelSource)};
    cl_program program = clCreateProgramWithSource(context, knlcount, knlstrList, knlsizeList, &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateProgramWithSource failed");

    // std::string buildopt = "-cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required";
    std::string buildopt = "-cl-intel-greater-than-4GB-buffer-required";
    err = clBuildProgram(program, 1, &device, buildopt.c_str(), nullptr, nullptr);
    if (err < 0)
    {
        size_t logsize = 0;
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsize);
        CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");

        std::vector<char> logbuf(logsize + 1, 0);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), nullptr);
        CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");
        printf("%s\n", logbuf.data());

        exit(1);
    }

    // GPU base test.
    {
        matmul_gpu_base(queue, program, "matmul_base", bufA, bufB, bufC, bufD, reference, M, N, K);
    }

    #define TILE_B 8
    // GPU matmul tile optimization
    if (M >= 8 && N >= 32)
    {
        #define TILE_OFM 1
        size_t gws[3] = {M * N / TILE_B / TILE_OFM, 1, 1};
        size_t lws[3] = {16, 1, 1};
        clReleaseMemObject(bufD);
        bufD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * M * N, D.data(), nullptr);
        matmul_gpu_bf_tile(queue, program, "matmul_bf_tile", bufA, bufB, bufC, bufD, reference, M, N, K, gws, lws);
    }

    // GPU matmul tile optimization 2
    if (N % 8 == 0 && K % 32 == 0)
    {
        size_t gws[3] = {M * N / TILE_B, 1, 1};
        size_t lws[3] = {8, 1, 1};
        clReleaseMemObject(bufD);
        bufD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * M * N, D.data(), nullptr);
        matmul_gpu_bf_tile(queue, program, "matmul_bf_tile_2", bufA, bufB, bufC, bufD, reference, M, N, K, gws, lws);
        // matmul_gpu_bf_tile(queue, program, "matmul_bf_tile_3", bufA, bufB, bufC, bufD, reference, M, N, K, gws, lws);
    }

    // GPU matmul XMX optimization
    if (N % 8 == 0 && K % 32 == 0 && M >= 8 && M%8 == 0)
    {
        // print_matrix(A, B, C, M, N, K);
        size_t gws[3] = {M * N / TILE_B, 1, 1};
        size_t lws[3] = {8, 1, 1};

        clReleaseMemObject(bufD);
        bufD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * M * N, D.data(), nullptr);
        matmul_gpu_bf_xmx(queue, program, "matmul_bf_xmx_1", bufA, bufB, bufC, bufD, reference, M, N, K, gws, lws);

        clReleaseMemObject(bufD);
        bufD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * M * N, D.data(), nullptr);
        matmul_gpu_bf_xmx(queue, program, "matmul_bf_xmx_2", bufA, bufB, bufC, bufD, reference, M, N, K, gws, lws);

        clReleaseMemObject(bufD);
        bufD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * M * N, D.data(), nullptr);
        matmul_gpu_bf_xmx(queue, program, "matmul_bf_xmx_3", bufA, bufB, bufC, bufD, reference, M, N, K, gws, lws);

        clReleaseMemObject(bufD);
        bufD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * M * N, D.data(), nullptr);
        #define BLOCK_NUM 4
        if (M >= TILE_B * BLOCK_NUM && N >= 8 && M % (TILE_B * BLOCK_NUM) == 0)
        {
            gws[0] = M * N / TILE_B / BLOCK_NUM;
            lws[0] = 8;
            matmul_gpu_bf_xmx(queue, program, "matmul_bf_xmx_4", bufA, bufB, bufC, bufD, reference, M, N, K, gws, lws);
        }
        #undef BLOCK_NUM

        #define SUBGROUP_TILE_B 2
        gws[0] = M * N / TILE_B;
        lws[0] = 8 * SUBGROUP_TILE_B;
        clReleaseMemObject(bufD);
        bufD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * M * N, D.data(), nullptr);
        matmul_gpu_bf_xmx(queue, program, "matmul_bf_xmx_5", bufA, bufB, bufC, bufD, reference, M, N, K, gws, lws);
        #undef SUBGROUP_TILE_B


        gws[0] = M * N / TILE_B * 2;
        lws[0] = 16;
        matmul_gpu_bf_xmx(queue, program, "matmul_bf_xmx_6", bufA, bufB, bufC, bufD, reference, M, N, K, gws, lws);

        gws[0] = M * N / TILE_B * 2;
        lws[0] = 16;
        clReleaseMemObject(bufD);
        bufD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * M * N, D.data(), nullptr);
        std::vector<int> reference_zero(M * K, 0);
        matmul_gpu_bf_xmx(queue, program, "matmul_bf_xmx_peak", bufA, bufB, bufC, bufD, reference_zero, M, N, K, gws, lws);

    }

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufD);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main(int argc, char **argv)
{
    int cnt = 1;
    int th_cnt = 1;
    if (argc < 4)
    {
        std::cout << "app_name <gpu_id> <M> <N> <K>" << std::endl;
        return 0;
    }
    int gpu_id = atoi(argv[1]);
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int K = atoi(argv[4]);
    cl_device_id device_id = choose_ocl_device(gpu_id, {"cl_intel_subgroups", "cl_intel_subgroup_matrix_multiply_accumulate", "cl_khr_integer_dot_product "});

    std::cout << std::endl
              << std::endl;

    run_matmul_test(device_id, M, N, K);

    return 1;
}
