# Test GPU data transfer bandwidth for Multi_GPUs

## How to build

    mkdir build && cd build && cmake .. && make


## How to run

    gpu_bandwidth_test <gpu_0 id> <gpu_1 id> <mem_size MB>

    For example:
    # ./ocl_matmul_test 0 1024 1024 2560
        GPU device number = 2
        Choose device: id = 0, name = Intel(R) Arc(TM) A770 Graphics

        matmul_cpu cost: 361.981 ms
        matmul_base cost: 40.5561 ms
        matmul_base check: SUCCESS

        matmul_bf_tile cost: 4.77017 ms
        matmul_bf_tile check: SUCCESS

        matmul_bf_tile_2 cost: 1.2533 ms
        matmul_bf_tile_2 check: SUCCESS

        matmul_bf_xmx_1 cost: 0.68158 ms
        matmul_bf_xmx_1 check: SUCCESS

        matmul_bf_xmx_2 cost: 0.220374 ms
        matmul_bf_xmx_2 check: SUCCESS

        matmul_bf_xmx_3 cost: 0.27579 ms
        matmul_bf_xmx_3 check: SUCCESS

        matmul_bf_xmx_4 cost: 0.162112 ms
        matmul_bf_xmx_4 check: SUCCESS

        matmul_bf_xmx_5 cost: 0.256863 ms
        matmul_bf_xmx_5 check: SUCCESS

        matmul_bf_xmx_6 cost: 0.294742 ms
        matmul_bf_xmx_6 check: FAIL(0.00683594)

        matmul_bf_xmx_peak cost: 0.0650976 ms
        matmul_bf_xmx_peak check: SUCCESS


