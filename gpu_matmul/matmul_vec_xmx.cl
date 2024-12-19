// --------------------------------- XMX compute vector multiply matrix
// M = 1
//
#define unroll_for __attribute__((opencl_unroll_hint)) for

#define dK 32   // int8 is 32
#define dX 4    // How many iter in BK
#define BK (dK*dX)  // 32*4=128
#define WG_SG_NUM_M 1
#define WG_SG_NUM_N 32
#define SG_BLOCK_NUM_M 1
#define SG_BLOCK_NUM_N 4

#define XMX_M 1

#define SG_OUTPUT_M (SG_BLOCK_NUM_M * XMX_M)   //  1*1=1
#define SG_OUTPUT_N (SG_BLOCK_NUM_N * 8)   // 4*8=32

#define WG_OUTPUT_M (WG_SG_NUM_M * SG_OUTPUT_M)   // 1*1=1
#define WG_OUTPUT_N (WG_SG_NUM_N * SG_OUTPUT_N)   // 32*32=1024

#define XMX_INPUT_BLOCK_SIZE (1 * dK)  // 1*32=32
#define XMX_WEIGHT_BLOCK_SIZE (8 * dK)     // 8*32=256 

#define SG_INPUT_BLOCK_SIZE (SG_OUTPUT_M  * dK)   // 1*32=32
#define SG_WEIGHT_BLOCK_SIZE (SG_OUTPUT_N * dK)  // 32*32=1024

// input data is light, so we can copy all input data into local memory by the first wi of one thread
#define SG_COPY_INPUT_BLOCK_SIZE (SG_OUTPUT_M * BK / WG_SG_NUM_N)   // 1*256/32=8

// weight data is heavy but cannot be shared, so need not copy weight data into local memory at all;
#define SG_COPY_WEIGHT_BLOCK_SIZE (SG_OUTPUT_N * BK / WG_SG_NUM_M)  // 32*256/1=8192
#define SG_COPY_WEIGHT_BLOCK_LINE (SG_COPY_WEIGHT_BLOCK_SIZE / 32)  // 8192/32=256
#define WI_COPY_WEIGHT_BLOCK_SIZE (SG_COPY_WEIGHT_BLOCK_SIZE / 8)   // 8192/8=1024  
#define WI_COPY_WEIGHT_BLOCK_LINE (WI_COPY_WEIGHT_BLOCK_SIZE / 32)  // 1024/32=32

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_vec_xmx_1(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint SIMD = (uint)get_sub_group_size();

    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
   
    uint sglid = (uint)get_sub_group_local_id();
    uint sg_c = get_local_id(0);
    uint sg_n = get_local_id(1);
    uint sg_m = get_local_id(2);
    uint sg_id = get_sub_group_id(); 
    
    // __local char in_slm[WG_OUTPUT_M * BK]; // 1*256=256
    // __local char wei_slm[WG_OUTPUT_N * BK];// 32*256=8192

    #if WI_COPY_WEIGHT_BLOCK_LINE==0
    #error "WI_COPY_WEIGHT_BLOCK_LINE is 0"
    #endif

    int result[SG_BLOCK_NUM_M][SG_BLOCK_NUM_N];
    {
        uint bias_offset = (wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M) * N + (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N);
        unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
            unroll_for(uint n = 0; n < SG_BLOCK_NUM_N; n++) {
                result[m][n] = (int)(biases[bias_offset + n * 8 + sg_c]);
            }
        }
    }

    #if SG_COPY_INPUT_BLOCK_SIZE != 8 && SG_COPY_INPUT_BLOCK_SIZE != 16 && SG_COPY_INPUT_BLOCK_SIZE != 32
    //#error "SG_COPY_INPUT_BLOCK_SIZE is not 8, 16 or 32"
    #endif

    unroll_for(uint ki = 0; ki < K; ki += BK) {
        // Copy input data into local memory
        #if 0
        if(sg_c == 0){
        // if(0){
            uint local_input_offset = sg_n *  SG_COPY_INPUT_BLOCK_SIZE; 
            uint input_offset = sg_n * SG_COPY_INPUT_BLOCK_SIZE + ki;
            #if SG_COPY_INPUT_BLOCK_SIZE == 4
                *(__local int *)(in_slm + local_input_offset) = *(__global int *)(input + input_offset);
            #elif SG_COPY_INPUT_BLOCK_SIZE == 8
                *(__local int2 *)(in_slm + local_input_offset) = *(__global int2 *)(input + input_offset);
            #elif SG_COPY_INPUT_BLOCK_SIZE == 16
                *(__local int4 *)(in_slm + local_input_offset) = *(__global int4 *)(input + input_offset);
            #elif SG_COPY_INPUT_BLOCK_SIZE == 32
                *(__local int8 *)(in_slm + local_input_offset) = *(__global int8 *)(input + input_offset);
            #endif
        }
        #endif
        // Copy weight data into local memory
        #if 0
        {
            uint local_wei_index = sg_n * SG_BLOCK_NUM_N * dX + sg_m * WI_COPY_WEIGHT_BLOCK_LINE;
            uint local_wei_offset = local_wei_index * XMX_WEIGHT_BLOCK_SIZE;// + sg_c * dK; // block_write
            uint n = wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE / dX) * 8 + sg_c;
            uint wei_offset =  n * K + ki + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE * dK) % BK;
            unroll_for(uint i = 0; i < WI_COPY_WEIGHT_BLOCK_LINE; i++) {
                int8 wei_cols_data =  *(__global int8 *)(weights + wei_offset + i * dK);
                intel_sub_group_block_write8((__local uint*)(wei_slm + local_wei_offset + i * XMX_WEIGHT_BLOCK_SIZE), as_uint8(wei_cols_data));
            }
        }
        #endif

        //barrier(CLK_LOCAL_MEM_FENCE);
        //const __local char* local_input_ptr = (const __local char *)(in_slm);

        //if(sg_c==0)
        //    printf("gid=%d, sg_id=%d, sg_c=%d, sg_n=%d, sg_m=%d, ki=%d, wei_offset=%d\n", gid, sg_id, sg_c, sg_n, sg_m, ki, wei_offset);
        __attribute__((opencl_unroll_hint(1)))
        for(int kii = 0; kii < BK; kii += dK) {
            int in_vec[SG_BLOCK_NUM_M];
            int8 wei_vec[SG_BLOCK_NUM_N];
            
            uint wei_offset =  (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N) * K + (ki + kii)* SG_OUTPUT_N;
            const __global char* global_wei_ptr  = (const __global char *)(weights  + wei_offset);

            unroll_for(int m = 0; m < SG_BLOCK_NUM_M; m++) {
                // in_vec[m] = as_int(intel_sub_group_block_read((__local uint*)(local_input_ptr + (kii + m * dX)* XMX_INPUT_BLOCK_SIZE)));
                in_vec[m] = as_int(intel_sub_group_block_read((__global uint*)(input + ki + kii + m * XMX_INPUT_BLOCK_SIZE)));
            }

            unroll_for(int n = 0; n < SG_BLOCK_NUM_N; n++) {
                // wei_vec[n] =  *(__global int8 *)(global_wei_ptr + sg_c* K + n * 8 * K);
                wei_vec[n] = as_int8(intel_sub_group_block_read8((__global uint*)(global_wei_ptr + n * XMX_WEIGHT_BLOCK_SIZE)));
            }

            unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
                unroll_for(uint n =0; n < SG_BLOCK_NUM_N; n++) {
                    result[m][n] = intel_sub_group_i8_i8_matrix_mad_k32(in_vec[m], wei_vec[n], result[m][n]);
                }
            }
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __global int *dst_ptr = output + (wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M) * N + (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N) + sg_c;
    unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
        unroll_for(uint n = 0; n < SG_BLOCK_NUM_N; n++) {
            dst_ptr[n * 8] = result[m][n];
        }
    }
}