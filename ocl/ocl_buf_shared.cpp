#include <iostream>
#include <string>
#include <vector>
#include <CL/cl.h>

cl_device_id choose_ocl_device(size_t id)
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
            std::cout << "Device: " << deviceName; // << std::endl;
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
                if (devices_vec.size() == id)
                {
                    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
                    std::cout << "Chosen device: " << deviceName << std::endl;
                }
                devices_vec.push_back(device);
            }
        }
    }
    if (devices_vec.size() > id)
        return devices_vec[id];
    return 0;
}

class oclStruct
{
public:
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem clbuf;
    void deinit()
    {
        clReleaseMemObject(clbuf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }
};

void ocl_share_cpu_mem_across_device()
{
    // Create host buffer
    size_t N = 16;
    std::vector<float> host_buf(N * N, 1.0f);
    size_t device_num = 2;

    oclStruct ocl_struct[2];
    for (size_t i = 0; i < device_num; i++)
    {
        // Choose device with OpenCL graphics
        ocl_struct[i].device = choose_ocl_device(i);
        ocl_struct[i].context = clCreateContext(nullptr, 1, &ocl_struct[i].device, nullptr, nullptr, nullptr);
        cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        ocl_struct[i].queue = clCreateCommandQueueWithProperties(ocl_struct[i].context, ocl_struct[i].device, props, nullptr);
        ocl_struct[i].clbuf = clCreateBuffer(ocl_struct[i].context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * N * N, host_buf.data(), nullptr);
    }

    const char *kernel_source = R"(
        __kernel void matrix_add(__global float* A, __global float* B) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int index = i * get_global_size(1) + j;
            A[index] += B[index];
            // printf("index = %d, res = %f\n", index, A[index]);
        }
    )";

    for (size_t i = 0; i < device_num; i++)
    {
        ocl_struct[i].program = clCreateProgramWithSource(ocl_struct[i].context, 1, &kernel_source, nullptr, nullptr);
        auto ret = clBuildProgram(ocl_struct[i].program, 1, &ocl_struct[i].device, nullptr, nullptr, nullptr);
        if (ret == CL_BUILD_PROGRAM_FAILURE)
        {
            printf("Could not build Kernel!\n");
            // Determine the size of the log
            size_t log_size;
            printf(" ret: %i\n", clGetProgramBuildInfo(ocl_struct[i].program, ocl_struct[i].device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));

            // Allocate memory for the log
            char *log = (char *)malloc(log_size);

            // Get the log
            printf(" ret: %i\n", clGetProgramBuildInfo(ocl_struct[i].program, ocl_struct[i].device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));

            // Print the log
            printf(" ret-val: %i\n", ret);
            printf("%s\n", log);
            free(log);
        }

        ocl_struct[i].kernel = clCreateKernel(ocl_struct[i].program, "matrix_add", nullptr);

        clEnqueueWriteBuffer(ocl_struct[i].queue, ocl_struct[i].clbuf, CL_TRUE, 0, N * N, host_buf.data(), 0, nullptr, nullptr);
        // auto err  = clEnqueueMigrateMemObjects(ocl_struct[i].queue, 1, &ocl_struct[i].clbuf, 0, 0, nullptr, nullptr);
        // ocl_struct[i].clbuf = clCreateBuffer(ocl_struct[i].context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * N * N, host_buf.data(), nullptr);
        clSetKernelArg(ocl_struct[i].kernel, 0, sizeof(cl_mem), &ocl_struct[i].clbuf);
        clSetKernelArg(ocl_struct[i].kernel, 1, sizeof(cl_mem), &ocl_struct[i].clbuf);
        size_t global_work_size[2] = {N, N};
        clEnqueueNDRangeKernel(ocl_struct[i].queue, ocl_struct[i].kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
        clFlush(ocl_struct[i].queue);
        clFinish(ocl_struct[i].queue);
        clEnqueueReadBuffer(ocl_struct[i].queue, ocl_struct[i].clbuf, CL_TRUE, 0, sizeof(float) * N * N, host_buf.data(), 0, nullptr, nullptr);

        // Read the output buffer back to the host
        std::vector<float> cl_result(N * N);
        clEnqueueReadBuffer(ocl_struct[i].queue, ocl_struct[i].clbuf, CL_TRUE, 0, sizeof(float) * N * N, cl_result.data(), 0, nullptr, nullptr);

        std::cout << std::endl
                  << "result " << i << ":" << std::endl;
        for (size_t j = 0; j < 16; j++)
        {
            std::cout << "\tres[" << j << "]=" << cl_result[j] << std::endl;
        }
    }

    for (size_t i = 0; i < device_num; i++)
    {
        ocl_struct[0].deinit();
    }
}

int main(int argc, char **argv)
{
    ocl_share_cpu_mem_across_device();
    return 1;
}