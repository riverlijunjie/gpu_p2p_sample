# Apply cl_event to sync command queue across gpu devices


## Build:
   mkdir build && cd build && cmake .. && make

## Run:
   ./ocl_event

    
    Choose device: Intel(R) Arc(TM) A770 Graphics
    Choose device: Intel(R) Arc(TM) A770 Graphics
    Choose device: Intel(R) Arc(TM) A770 Graphics
    Copy time: 14.9521 ms, size = 131072 KB
    Add time: 5.13328 ms, size = 131072KB
    Use clFinish to sync queues between different gpu devices: SUCCESS(0.000)

    Choose device: Intel(R) Arc(TM) A770 Graphics
    Choose device: Intel(R) Arc(TM) A770 Graphics
    Copy time: 0.132973 ms, size = 131072 KB
    Add time: 4.58097 ms, size = 131072KB
    Use event to sync queues between different gpu devices: FAIL(0.982)




