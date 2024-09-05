#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <map>

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

static std::map<int, const char *> oclChannelOrder =
    {
        {0x10B0, "CL_R"},
        {0x10B1, "CL_A"},
        {0x10B2, "CL_RG"},
        {0x10B3, "CL_RA"},
        {0x10B4, "CL_RGB"},
        {0x10B5, "CL_RGBA"},
        {0x10B6, "CL_BGRA"},
        {0x10B7, "CL_ARGB"},
        {0x10B8, "CL_INTENSITY"},
        {0x10B9, "CL_LUMINANCE"},
        {0x10BA, "CL_Rx"},
        {0x10BB, "CL_RGx"},
        {0x10BC, "CL_RGBx"},
        {0x10BD, "CL_DEPTH"},
        {0x10BE, "CL_DEPTH_STENCIL"},
        {0x10BF, "CL_sRGB"},
        {0x10C0, "CL_sRGBx"},
        {0x10C1, "CL_sRGBA"},
        {0x10C2, "CL_sBGRA"},
        {0x10C3, "CL_ABGR"},
};

static std::map<int, const char *> oclChannelType =
    {
        {0x10D0, "CL_SNORM_INT8"},
        {0x10D1, "CL_SNORM_INT16"},
        {0x10D2, "CL_UNORM_INT8"},
        {0x10D3, "CL_UNORM_INT16"},
        {0x10D4, "CL_UNORM_SHORT_565"},
        {0x10D5, "CL_UNORM_SHORT_555"},
        {0x10D6, "CL_UNORM_INT_101010"},
        {0x10D7, "CL_SIGNED_INT8"},
        {0x10D8, "CL_SIGNED_INT16"},
        {0x10D9, "CL_SIGNED_INT32"},
        {0x10DA, "CL_UNSIGNED_INT8"},
        {0x10DB, "CL_UNSIGNED_INT16"},
        {0x10DC, "CL_UNSIGNED_INT32"},
        {0x10DD, "CL_HALF_FLOAT"},
        {0x10DE, "CL_FLOAT"},
        {0x10DF, "CL_UNORM_INT24"},
        {0x10E0, "CL_UNORM_INT_101010_2"},
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

#define CL_MEM_ALLOCATION_HANDLE_INTEL 0x10050

static int g_device_num = 0;
cl_device_id choose_ocl_device(size_t id = 0)
{
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    std::vector<cl_device_id> devices_vec;
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
                char deviceName[128];
                if (devices_vec.size() == id)
                {
                    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
                    std::cout << "Choose device: " << deviceName << std::endl;
                }
                devices_vec.push_back(device);
            }
        }
    }
    g_device_num = devices_vec.size();
    std::cout << "GPU device number = " << devices_vec.size() << ", choose device_id = "<< id << std::endl;
    if (devices_vec.size() > id)
        return devices_vec[id];
    return 0;
}

void run_queue_sync_test(bool is_event_sync_mode, size_t bytes_size, cl_device_id device_1, cl_device_id device_2)
{
    //cl_device_id device_1 = choose_ocl_device(0);
    //cl_device_id device_2 = choose_ocl_device(1);

    cl_context context_1 = clCreateContext(nullptr, 1, &device_1, nullptr, nullptr, nullptr);
    cl_context context_2 = clCreateContext(nullptr, 1, &device_2, nullptr, nullptr, nullptr);

    cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
    cl_command_queue queue_1 = clCreateCommandQueueWithProperties(context_1, device_1, props, nullptr);
    cl_command_queue queue_2 = clCreateCommandQueueWithProperties(context_2, device_2, props, nullptr);

    const int size = bytes_size;// 128 * 1024 * 1024; // 128MB
    std::vector<uint8_t> A(size, 0x3);
    std::vector<uint8_t> C(size, 0x7);
    cl_mem bufA = clCreateBuffer(context_1, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int8_t) * size, A.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context_2, CL_MEM_READ_WRITE, sizeof(int8_t) * size, nullptr, nullptr);
    cl_mem bufC = clCreateBuffer(context_2, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int8_t) * size, C.data(), nullptr);
    cl_mem bufD = clCreateBuffer(context_2, CL_MEM_READ_WRITE, sizeof(int8_t) * size, nullptr, nullptr);

    uint64_t fd;
    cl_int err = clGetMemObjectInfo(bufB, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(fd), &fd, NULL);
    CHECK_OCL_ERROR_EXIT(err, "clGetMemObjectInfo failed");

    cl_mem_properties extMemProperties[] = {
        (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
        (cl_mem_properties)fd,
        0};
    cl_mem shared_mem = clCreateBufferWithProperties(context_1, extMemProperties, 0, size, NULL, &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateBufferWithProperties failed");

    // Define the kernel source code
    const char *kernelSource = R"(
        __kernel void matrix_add(__global const char* A, __global const char* B, __global char* C) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            C[index] = A[index] + B[index];
            // printf("%d + %d = %d\n",A[index],B[index],C[index]);
        }
    )";
    // Create a program from the kernel source
    cl_uint knlcount = 1;
    const char *knlstrList[] = {kernelSource};
    size_t knlsizeList[] = {strlen(kernelSource)};
    cl_program program = clCreateProgramWithSource(context_2, knlcount, knlstrList, knlsizeList, &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateProgramWithSource failed");

    std::string buildopt = "-cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required";
    err = clBuildProgram(program, 1, &device_2, buildopt.c_str(), nullptr, nullptr);
    if (err < 0)
    {
        size_t logsize = 0;
        err = clGetProgramBuildInfo(program, device_2, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsize);
        CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");

        std::vector<char> logbuf(logsize + 1, 0);
        err = clGetProgramBuildInfo(program, device_2, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), nullptr);
        CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");
        printf("%s\n", logbuf.data());

        exit(1);
    }
    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "matrix_add", &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateKernel failed");

    std::vector<uint8_t> result(size, 0x0);
    cl_event event;
    const auto start_1 = std::chrono::high_resolution_clock::now();
    if (is_event_sync_mode)
    {
        err = clEnqueueCopyBuffer(queue_1, bufA, shared_mem, 0, 0, size, 0, nullptr, &event);
        clWaitForEvents(1, &event);
    }
    else
    {
        err = clEnqueueCopyBuffer(queue_1, bufA, shared_mem, 0, 0, size, 0, nullptr, nullptr);
        clFinish(queue_1);
    }
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueCopyBuffer failed");

    const auto end_1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_1 = end_1 - start_1;
    std::cout << "Copy time: " << elapsed_1.count() << " ms, size = " << size / 1024 << " KB, BandWidth = " << size/1024.0/1024/1024/elapsed_1.count()*1000 << " GB/s" << std::endl;


    err = clEnqueueReadBuffer(queue_2, bufD, CL_TRUE, 0, sizeof(int8_t) * size, result.data(), 0, nullptr, nullptr);

    // Set the kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufD);
    // Execute the kernel
    size_t globalSize[1] = {size};
    const auto start_2 = std::chrono::high_resolution_clock::now();
    if (is_event_sync_mode)
    {
        err = clEnqueueNDRangeKernel(queue_2, kernel, 1, nullptr, globalSize, nullptr, 1, &event, nullptr);
        clFinish(queue_2);
    }
    else
    {
        err = clEnqueueNDRangeKernel(queue_2, kernel, 1, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue_2);
    }
    const auto end_2 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_2 = end_2 - start_2;
    std::cout << "Add time: " << elapsed_2.count() << " ms, size = " << size / 1024 << "KB" << std::endl;
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");

    err = clEnqueueReadBuffer(queue_2, bufD, CL_TRUE, 0, sizeof(int8_t) * size, result.data(), 0, nullptr, nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");

    size_t cnt = 0;
    for (int i = 0; i < size; i += 1)
    {
        if (A[i] + C[i] != result[i])
        {
            cnt += 1;
        }
    }

    if (is_event_sync_mode)
    {
        printf("Use event to sync queues between different gpu devices: %s(%.3f)\n\n", cnt == 0 ? "SUCCESS" : "FAIL", 1.0*cnt/size);
    }
    else
    {
        printf("Use clFinish to sync queues between different gpu devices: %s(%.3f)\n\n", cnt == 0 ? "SUCCESS" : "FAIL", 1.0*cnt/size);
    }

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufD);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue_1);
    clReleaseCommandQueue(queue_2);
    clReleaseContext(context_1);
    clReleaseContext(context_2);
}

int main(int argc, char **argv)
{
    int cnt = 1;
    if (argc < 4) {
         std::cout << "app_name <gpu_id> <gpu_id> <mem_size MB> <loop_cnt>" << std::endl;
         return 0;
    }
    int gpu_0 = atoi(argv[1]);
    int gpu_1 = atoi(argv[2]);
    int bytes = atoi(argv[3]) * 1024 * 1024;
    
    if(argc==5)
        cnt = atoi(argv[4]);

    const int data_size[] = {1, 4, 128, 512};
    cl_device_id device_1 = choose_ocl_device(gpu_0);
    cl_device_id device_2 = choose_ocl_device(gpu_1);
    std::cout << std::endl << std::endl;
    for(int k = 0; k < cnt; k++) {
        // Test queue sync between different contexts
        std::cout << "Loop count: " << k << " ......................" << std::endl;
	std::cout << "GPU index: " << gpu_0 << "-->" << gpu_1 << std::endl;
        run_queue_sync_test(0, bytes, device_1, device_2); // manual sync(clFinish) will sucess
        std::cout << "GPU index: " << gpu_1 << "-->" << gpu_0 << std::endl;
        run_queue_sync_test(0, bytes, device_2, device_1); // manual sync(clFinish) will sucess
        // run_queue_sync_test(1, bytes, device_1, device_2); // cl_event sync will fail
        std::cout << std::endl;
    }

    return 1;
}
