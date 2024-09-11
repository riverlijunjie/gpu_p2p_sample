# Test OCL P2P data transfer bandwidth for Multi_GPUs

## How to build

    mkdir build && cd build && cmake .. && make


## How to run

    ocl_p2p_bandwith_test <gpu_0 id> <gpu_1 id> <mem_size MB> <loop cnt>

    For example:
        # ./ocl_p2p_bandwith_test 0 1 16 2

            Choose device: Intel(R) Arc(TM) A770 Graphics
            GPU device number = 2, choose device_id = 0
            Choose device: Intel(R) Arc(TM) A770 Graphics
            GPU device number = 2, choose device_id = 1


            Loop count: 0 ......................
            GPU index: 0-->1
            Copy time: 2.82707 ms, size = 16384 KB, BandWidth = 5.52692 GB/s
            GPU P2P between different gpu devices: SUCCESS(0.000)
            Add time: 1.37533 ms, size = 16384KB
            Use clFinish to sync queues between different gpu devices: SUCCESS(0.000)

            GPU index: 1-->0
            Copy time: 2.11167 ms, size = 16384 KB, BandWidth = 7.39934 GB/s
            GPU P2P between different gpu devices: SUCCESS(0.000)
            Add time: 1.45484 ms, size = 16384KB
            Use clFinish to sync queues between different gpu devices: SUCCESS(0.000)


            Loop count: 1 ......................
            GPU index: 0-->1
            Copy time: 2.13872 ms, size = 16384 KB, BandWidth = 7.30576 GB/s
            GPU P2P between different gpu devices: SUCCESS(0.000)
            Add time: 1.08016 ms, size = 16384KB
            Use clFinish to sync queues between different gpu devices: SUCCESS(0.000)

            GPU index: 1-->0
            Copy time: 2.31129 ms, size = 16384 KB, BandWidth = 6.76029 GB/s
            GPU P2P between different gpu devices: SUCCESS(0.000)
            Add time: 0.974006 ms, size = 16384KB
            Use clFinish to sync queues between different gpu devices: SUCCESS(0.000)


