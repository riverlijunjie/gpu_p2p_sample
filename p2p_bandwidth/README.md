# Test OCL P2P data transfer bandwidth for Multi_GPUs

## How to build

    mkdir build && cd build && cmake .. && make


## How to run

    ocl_p2p_bandwith_test <gpu_0 id> <gpu_1 id> <mem_size MB> <loop cnt>

    For example:
        # ./ocl_p2p_bandwith_test 0 1 16 2

        Choose device: Intel(R) Arc(TM) A770 Graphics
        GPU device number = 4, choose device_id = 0
        Choose device: Intel(R) Arc(TM) A770 Graphics
        GPU device number = 4, choose device_id = 1


        Loop count: 0 ......................
        GPU index: 0-->1
        Copy time: 2.9693 ms, size = 16384 KB, BandWidth = 5.26219 GB/s
        Add time: 1.36095 ms, size = 16384KB
        Use clFinish to sync queues between different gpu devices: SUCCESS(0.000)

        GPU index: 1-->0
        Copy time: 2.33331 ms, size = 16384 KB, BandWidth = 6.6965 GB/s
        Add time: 1.343 ms, size = 16384KB
        Use clFinish to sync queues between different gpu devices: SUCCESS(0.000)


        Loop count: 1 ......................
        GPU index: 0-->1
        Copy time: 2.33303 ms, size = 16384 KB, BandWidth = 6.69729 GB/s
        Add time: 0.929862 ms, size = 16384KB
        Use clFinish to sync queues between different gpu devices: SUCCESS(0.000)

        GPU index: 1-->0
        Copy time: 2.34268 ms, size = 16384 KB, BandWidth = 6.66971 GB/s
        Add time: 0.9282 ms, size = 16384KB
        Use clFinish to sync queues between different gpu devices: SUCCESS(0.000)


