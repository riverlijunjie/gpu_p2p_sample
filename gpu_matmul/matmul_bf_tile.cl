#include "../common.cl"
#include "../sub_group_block_read.cl"
#include "../sub_group_block_write.cl" 
        
#define TILE_B 8
#define TILE_OFM 1
#define TILE_IFM 1
#define TILE_K 1
#define TILE_K_OFM  1 /*TILE_OFM*TILE_K*/

#define INPUT0_TYPE char
#define ACCUMULATOR_TYPE int
#define FILTER_TYPE char
#define BIAS_TYPE char
#define ACTIVATION_TYPE int
#define OUTPUT_TYPE int

// Need check below value
#define INPUT0_OFFSET 0
#define USE_BLOCK_WRITE 1

#define unroll_for __attribute__((opencl_unroll_hint)) for
// Macros for vectorized types.
#define INPUT_VEC_TYPE             MAKE_VECTOR_TYPE(INPUT0_TYPE, TILE_IFM)
#define ACCUMULATOR_VEC_TYPE       MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define FILTER_VEC_TYPE            MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_K_OFM)
#define FILTER_PACKED_VEC_TYPE     MAKE_VECTOR_TYPE(FILTER_TYPE, TILE_K_OFM)
#define BIAS_VEC_TYPE              MAKE_VECTOR_TYPE(BIAS_TYPE, TILE_OFM)
#define OUTPUT_VEC_TYPE            MAKE_VECTOR_TYPE(OUTPUT_TYPE, TILE_OFM)
#define ACTIVATION_VEC_TYPE        MAKE_VECTOR_TYPE(ACTIVATION_TYPE, TILE_OFM)
#define TO_OUTPUT_VEC_TYPE(x)      CAT(convert_, OUTPUT_VEC_TYPE)(x)
#define TO_ACTIVATION_VEC_TYPE(x)  CAT(convert_, ACTIVATION_VEC_TYPE)(x)
#define TO_FILTER_VEC_TYPE(x)      CAT(convert_, FILTER_VEC_TYPE)(x)
#define TO_ACCUMULATOR_VEC_TYPE(x) CAT(convert_, ACCUMULATOR_VEC_TYPE)(x)

#define INPUT_BLOCK_READ(ptr, offset)        BLOCK_READN(INPUT0_TYPE, TILE_IFM, ptr, offset)
#define FILTER_BLOCK_READ(ptr, offset)       BLOCK_READN(FILTER_TYPE, TILE_K_OFM, ptr, offset)
#define BIAS_BLOCK_READ(ptr, offset)         BLOCK_READN(BIAS_TYPE, TILE_OFM, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, TILE_OFM, ptr, offset, val)

#define AS_TYPE(type, val) CAT(as_, type)(val)

void print_wi_info_(uint out_b, uint out_f) {
    printf("global_id = (%d,%d,%d), local_id = (%d,%d,%d), group_id = (%d,%d,%d), subgroup_id=%d, subgroup_size=%d, subgroup_local_id=%d, out_b = %d, out_f = %d\n",
                get_global_id(0), get_global_id(1),get_global_id(2),
                get_local_id(0), get_local_id(1),get_local_id(2),
                get_group_id(0), get_group_id(1),get_group_id(2),
                get_sub_group_id(),get_sub_group_size(),
                get_sub_group_local_id(), out_b, out_f);
}

// M=1024, N=1024, K=2560
// 8x16, SIMD=16
// 4.8 ms
// __attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul_bf_tile(__global const char* input, __global const char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    /*
    printf("global_id = (%d,%d,%d), local_id = (%d,%d,%d), group_id = (%d,%d,%d), subgroup_id=%d, subgroup_size=%d, subgroup_local_id=%d\n",
                get_global_id(0), get_global_id(1),get_global_id(2),
                get_local_id(0), get_local_id(1),get_local_id(2),
                get_group_id(0), get_group_id(1),get_group_id(2),
                get_sub_group_id(),get_sub_group_size(),
                get_sub_group_local_id());
    */
    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (CEIL_DIV(N, TILE_OFM * SIMD));
    uint sg_block_b = sg_id / CEIL_DIV(N, TILE_OFM * SIMD);
    uint out_f = sg_block_f * (TILE_OFM * SIMD);
    uint out_b = sg_block_b * TILE_B;

    ACCUMULATOR_VEC_TYPE acc[TILE_B] = { };
    INPUT_VEC_TYPE       in_0[TILE_B] = { };
    FILTER_VEC_TYPE      wei = 0;

    uint input_offset = out_b * K;
    uint weights_offset = out_f * K;

    // Main loop
    uint iterations = K / (TILE_IFM * SIMD);
    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        // Load input.
        // printf("iter=%d: (gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): input_offset = %d\n",ni, gid, SIMD, sglid, out_b, out_f, input_offset);
        unroll_for(uint bi=0; bi<TILE_B; bi++) {
            in_0[bi] = INPUT_BLOCK_READ(input, input_offset);
            // in_0[bi] = input[input_offset + sglid];
            input_offset += K;
        }
        input_offset += TILE_IFM * SIMD - K * TILE_B;
        // ACCUMULATOR_VEC_TYPE acc_tmp[TILE_B] = { };
        // printf("iter=%d: (gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): weights_offset = %d\n",ni, gid, SIMD, sglid, out_b, out_f, weights_offset);
        unroll_for(uint ki = 0; ki < (TILE_IFM * SIMD); ++ki) {
            //wei = TO_FILTER_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_offset));
            wei = TO_FILTER_VEC_TYPE(weights[weights_offset + sglid*K]);
            unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
                INPUT0_TYPE in_val = sub_group_shuffle(((INPUT0_TYPE*)(&in_0[bi]))[ki / SIMD], ki % SIMD);
                acc[bi] += in_val * ((ACCUMULATOR_TYPE*)(&wei))[0];
            }
            weights_offset += 1;
        }
    } // Main loop

    // BIAS
    BIAS_VEC_TYPE bias[TILE_B] = {};
    uint bias_offset = out_f  + out_b * N;
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        bias[bi] = BIAS_BLOCK_READ(biases, bias_offset );
        // bias[bi] = biases[bias_offset + sglid];
        bias_offset += N;
    }

    OUTPUT_VEC_TYPE result[TILE_B] = { };
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        result[bi] = TO_OUTPUT_VEC_TYPE(acc[bi] + bias[bi]);
    }

    // Write results
    uint output_offset = out_f  + out_b * N ;
    // printf("(gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): input_offset = %d\n", gid, SIMD, sglid, out_b, out_f, output_offset);
    unroll_for(uint bi = 0; bi < TILE_B; bi++) {
        OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);
        // output[output_offset + sglid] = result[bi];
        output_offset += N;
    }
    // kernel end
}


// M=1024, N=1024, K=2560
// 8x32, SIMD=8
// 1.2 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_tile_2(__global const char* input, __global const char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    #define BLOCK_K 32
    #define BLOCK_B 8
    #define BLOCK_F 8

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N/SIMD);
    uint sg_block_b = sg_id / (N/SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;

    // print_wi_info_(out_b, out_f);

    int acc[BLOCK_B] = { };
    char4  in_0[BLOCK_B] = { };
    char4  wei = { };

    // Main loop
    uint iterations = K / BLOCK_K;
    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        // Load input.
        // printf("iter=%d: (gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): input_offset = %d\n",ni, gid, SIMD, sglid, out_b, out_f, input_offset);
        uint input_offset = out_b * K + ni * BLOCK_K;
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            in_0[bi] = as_char4(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            // printf("in_0[%d:%d] = [%d,%d,%d,%d]\n", sglid, bi, in_0[bi][0], in_0[bi][1],in_0[bi][2],in_0[bi][3]);
            input_offset += K;
        }
        // barrier(CLK_LOCAL_MEM_FENCE);
        // ACCUMULATOR_VEC_TYPE acc_tmp[TILE_B] = { };
        // printf("iter=%d: (gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): weights_offset = %d\n",ni, gid, SIMD, sglid, out_b, out_f, weights_offset);
        unroll_for(uint ki = 0; ki < BLOCK_K; ki += 4) {
            uint weights_offset = out_f * K + sglid * K + ki + ni * BLOCK_K;
            wei=as_char4(*(const __global int*)(weights + weights_offset));
            unroll_for (uint bi = 0; bi < BLOCK_B; ++bi) {
                char4 in_val = intel_sub_group_shuffle(in_0[bi], (ki/4) % SIMD);
                // printf("sgid = %d, in_val[%d] = [%d,%d,%d,%d]\n", sglid, ki, in_val[0], in_val[1],in_val[2],in_val[3]);
                acc[bi] += in_val[0] * wei[0] + in_val[1] * wei[1] + in_val[2] * wei[2] + in_val[3] * wei[3];
                // Require __opencl_c_integer_dot_product_input_4x8bit
                //int temp = dot(in_val, wei);
            }
        }
    } // Main loop

    // BIAS
    char bias[TILE_B] = {};
    uint bias_offset = out_f  + out_b * N;
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        bias[bi] = biases[bias_offset + sglid];
        bias_offset += N;
    }

    int result[TILE_B] = { };
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        result[bi] = as_int(acc[bi] + bias[bi]);
    }

    // Write results
    uint output_offset = out_f  + out_b * N ;
    // printf("(gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): input_offset = %d\n", gid, SIMD, sglid, out_b, out_f, output_offset);
    unroll_for(uint bi = 0; bi < TILE_B; bi++) {
        // OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);
        intel_sub_group_block_write(output + output_offset, result[bi]);
        // output[output_offset + sglid] = result[bi];
        output_offset += N;
    }
    // kernel end
}


// M=1024, N=1024, K=2560
// 8x32 + EU tile 2x2
// 1.2 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_tile_3(__global const char* input, __global const char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    #define BLOCK_K 32
    #define BLOCK_B 8
    #define BLOCK_F 8

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N/SIMD);
    uint sg_block_b = sg_id / (N/SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;

    // print_wi_info_(out_b, out_f);

    int acc[BLOCK_B] = { };
    char4  in_0[BLOCK_B] = { };
    char4  wei = { };

    // Main loop
    uint iterations = K / BLOCK_K;
    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        // Load input.
        // printf("iter=%d: (gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): input_offset = %d\n",ni, gid, SIMD, sglid, out_b, out_f, input_offset);
        uint input_offset = out_b * K + ni * BLOCK_K;
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            in_0[bi] = as_char4(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            // printf("in_0[%d:%d] = [%d,%d,%d,%d]\n", sglid, bi, in_0[bi][0], in_0[bi][1],in_0[bi][2],in_0[bi][3]);
            input_offset += K;
        }
        // barrier(CLK_LOCAL_MEM_FENCE);
        // ACCUMULATOR_VEC_TYPE acc_tmp[TILE_B] = { };
        // printf("iter=%d: (gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): weights_offset = %d\n",ni, gid, SIMD, sglid, out_b, out_f, weights_offset);
        unroll_for(uint ki = 0; ki < BLOCK_K; ki += 4) {
            uint weights_offset = out_f * K + sglid * K + ki;
            wei=as_char4(*(const __global int*)(weights + weights_offset));
            unroll_for (uint bi = 0; bi < BLOCK_B; ++bi) {
                char4 in_val = intel_sub_group_shuffle(in_0[bi], (ki/4) % SIMD);
                // printf("sgid = %d, in_val[%d] = [%d,%d,%d,%d]\n", sglid, ki, in_val[0], in_val[1],in_val[2],in_val[3]);
                acc[bi] += in_val[0] * wei[0] + in_val[1] * wei[1] + in_val[2] * wei[2] + in_val[3] * wei[3];
            }
        }
    } // Main loop

    // BIAS
    char bias[TILE_B] = {};
    uint bias_offset = out_f  + out_b * N;
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        bias[bi] = biases[bias_offset + sglid];
        bias_offset += N;
    }

    int result[TILE_B] = { };
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        result[bi] = as_int(acc[bi] + bias[bi]);
    }

    // Write results
    uint output_offset = out_f  + out_b * N ;
    // printf("(gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): input_offset = %d\n", gid, SIMD, sglid, out_b, out_f, output_offset);
    unroll_for(uint bi = 0; bi < TILE_B; bi++) {
        intel_sub_group_block_write(output + output_offset, result[bi]);
        // output[output_offset + sglid] = result[bi];
        output_offset += N;
    }
    // kernel end
}
