# Test GPU data transfer bandwidth for Multi_GPUs

## How to build

    mkdir build && cd build && cmake .. && make


## How to run

    gpu_bandwidth_test <gpu_0 id> <gpu_1 id> <mem_size MB>

    For example:
        # ./gpu_bandwidth_test 0 1 256
            Choose device: Intel(R) Arc(TM) A770 Graphics
            GPU device number = 2, choose device_id = 0
            Choose device: Intel(R) Arc(TM) A770 Graphics
            GPU device number = 2, choose device_id = 1


            GPU index: 0-->1
            size = 262144 KB
                    GPU->GPU copy 0: time = 2.58848 ms, size = 262144 KB, BandWidth = 96.5819 GB/s
                    GPU->CPU copy 0: time = 37.0543 ms, size = 262144 KB, BandWidth = 6.74686 GB/s
                    GPU->GPU copy 1: time = 1.37214 ms, size = 262144 KB, BandWidth = 182.197 GB/s
                    GPU->CPU copy 1: time = 17.2166 ms, size = 262144 KB, BandWidth = 14.5209 GB/s
                    GPU->GPU copy 2: time = 1.31687 ms, size = 262144 KB, BandWidth = 189.844 GB/s
                    GPU->CPU copy 2: time = 17.0261 ms, size = 262144 KB, BandWidth = 14.6833 GB/s
            GPU->GPU bandwidh = 156.208 GB/s
            GPU->CPU bandwidh = 11.9837 GB/s
                    P2P copy 0 : time = 12.7973 ms, size = 262144 KB, BandWidth = 19.5354 GB/s
                    P2P copy 1 : time = 12.704 ms, size = 262144 KB, BandWidth = 19.6788 GB/s
                    P2P copy 2 : time = 12.6988 ms, size = 262144 KB, BandWidth = 19.6869 GB/s
            GPU P2P bandwidth: 19.6337 GB/s


