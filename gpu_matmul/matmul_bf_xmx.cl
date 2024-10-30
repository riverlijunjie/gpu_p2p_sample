#include "../common.cl"
#include "../sub_group_block_read.cl"
#include "../sub_group_block_write.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for

#define BLOCK_K 32
#define BLOCK_B 8
#define BLOCK_F 8

void print_output_int8(int8 result, uint sglid, uint it, uint sg_id) {
    printf("output[%d:%d]-%d = [%d,%d,%d,%d,%d,%d,%d, %d]\n", sg_id, sglid, it, result[0], result[1],result[2],result[3],result[4],result[5],result[6],result[7]);
}

void print_input_int8(int8 result, uint sglid) {
    char4 temp[8];
    unroll_for(uint i = 0; i < 8; i++) {
        temp[i] = as_char4(result[i]);
    }
    printf("input[%d] = [%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d]\n", sglid,
            temp[0][0], temp[0][1],temp[0][2],temp[0][3],
            temp[1][0], temp[1][1],temp[1][2],temp[1][3],
            temp[2][0], temp[2][1],temp[2][2],temp[2][3],
            temp[3][0], temp[3][1],temp[3][2],temp[3][3],
            temp[4][0], temp[4][1],temp[4][2],temp[4][3],
            temp[5][0], temp[5][1],temp[5][2],temp[5][3],
            temp[6][0], temp[6][1],temp[6][2],temp[6][3],
            temp[7][0], temp[7][1],temp[7][2],temp[7][3]);
}

void print_input_int4(int4 result, uint sglid) {
    char4 temp[4];
    unroll_for(uint i = 0; i < 4; i++) {
        temp[i] = as_char4(result[i]);
    }
    printf("input[%d] = [%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d]\n", sglid,
            temp[0][0], temp[0][1],temp[0][2],temp[0][3],
            temp[1][0], temp[1][1],temp[1][2],temp[1][3],
            temp[2][0], temp[2][1],temp[2][2],temp[2][3],
            temp[3][0], temp[3][1],temp[3][2],temp[3][3]);
}

void print_weight_int8(int result[8], uint sglid) {
    char4 temp[8];
    unroll_for(uint i = 0; i < 8; i++) {
        temp[i] = as_char4(result[i]);
    }
    printf("weight[%d] = [%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d]\n", sglid,
            temp[0][0], temp[0][1],temp[0][2],temp[0][3],
            temp[1][0], temp[1][1],temp[1][2],temp[1][3],
            temp[2][0], temp[2][1],temp[2][2],temp[2][3],
            temp[3][0], temp[3][1],temp[3][2],temp[3][3],
            temp[4][0], temp[4][1],temp[4][2],temp[4][3],
            temp[5][0], temp[5][1],temp[5][2],temp[5][3],
            temp[6][0], temp[6][1],temp[6][2],temp[6][3],
            temp[7][0], temp[7][1],temp[7][2],temp[7][3]);
}

void print_input_uint8(uint8 result, uint sglid) {
    char4 temp = as_char4(result[0]);
    char4 temp1=as_char4(result[1]);
    printf("uinput[%d] = [%d,%d,%d,%d,%d,%d,%d,%d]\n", sglid, temp[0], temp[1],temp[2],temp[3],temp1[0],temp1[1],temp1[2],temp1[3]);
}

void reference(__global char* input_base, __global char* weights_base, int K, uint sg_id) {
    int8 result = {};
    uint input_offset = 0;
    uint sglid = (uint)get_sub_group_local_id();
    unroll_for(uint b=0;b<8;b++) {
        result[b] = 0;
        unroll_for(uint f=0;f<32;f++) {
            result[b] += input_base[input_offset+f] * weights_base[f];
        }
        input_offset +=K;
    }
    printf("ref[%d:%d] = [%d,%d,%d,%d,%d,%d,%d,%d]\n", sg_id, sglid, result[0], result[1],result[2],result[3],result[4],result[5],result[6],result[7]);
}

void print_wi_info(uint out_b, uint out_f) {
    printf("global_id = (%d,%d,%d), local_id = (%d,%d,%d), group_id = (%d,%d,%d), subgroup_id=%d, subgroup_size=%d, subgroup_local_id=%d, out_b = %d, out_f = %d\n",
                get_global_id(0), get_global_id(1),get_global_id(2),
                get_local_id(0), get_local_id(1),get_local_id(2),
                get_group_id(0), get_group_id(1),get_group_id(2),
                get_sub_group_id(),get_sub_group_size(),
                get_sub_group_local_id(), out_b, out_f);
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_0(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N / SIMD);
    uint sg_block_b = sg_id / (N / SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;

    // print_wi_info(out_b, out_f);
    int8 result = 0;
    int8 a,b,c={};
    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int8 temp8;
        unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
            char4 temp;
            temp[0]=input[input_offset + sglid * 4];
            temp[1]=input[input_offset + sglid * 4 + 1];
            temp[2]=input[input_offset + sglid * 4 + 2];
            temp[3]=input[input_offset + sglid * 4 + 3];
            a[bi]= as_int(temp);
            input_offset += K;
        }
        uint weights_offset = out_f * K + sglid * K + ki;
        unroll_for(uint fi =0; fi < BLOCK_F; fi++) {
            char4 temp;
            temp[0]=weights[weights_offset + fi * 4];
            temp[1]=weights[weights_offset + fi * 4 + 1];
            temp[2]=weights[weights_offset + fi * 4 + 2];
            temp[3]=weights[weights_offset + fi * 4 + 3];
            b[fi]=as_int(temp);
        }
        // barrier(CLK_LOCAL_MEM_FENCE);
        temp8 = intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
        result += temp8;
        // print_input_int8(a, sglid);
        // print_input_int8(b, sglid);
        // print_output_int8(temp8, sglid, ki/BLOCK_K, sg_id);
        // reference(input + out_b * K + ki, weights + out_f * K + sglid * K + ki, K, sg_id);
    }
    // print_output_int8(result, sglid, 2, sg_id);

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
        output[output_offset + sglid] = result[bi];
        output_offset += N;
    }
}
// M=1024, N=1024, K=2560
// Base version, without memory share in subgroup
// 0.681392 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_1(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N / SIMD);
    uint sg_block_b = sg_id / (N / SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;

    // print_wi_info(out_b, out_f);
    int8 result = 0;
    int8 a,b,c={};
    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
            char4 temp;
            unroll_for(uint it = 0; it < 4; it++) {
                temp[it] = input[input_offset + sglid * 4 + it];
            }
            a[bi]= as_int(temp);
            input_offset += K;
        }
        uint weights_offset = out_f * K + sglid * K + ki;
        unroll_for(uint fi =0; fi < BLOCK_F; fi++) {
            char4 temp;
            unroll_for(uint it = 0; it < 4; it++) {
                temp[it]=weights[weights_offset + fi * 4 + it];
            }
            b[fi]=as_int(temp);
        }
        result += intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
        // print_input_int8(a, sglid);
        // print_input_int8(b, sglid);
        // print_output_int8(temp8, sglid, ki/BLOCK_K, sg_id);
    }

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
        output[output_offset + sglid] = result[bi];
        output_offset += N;
    }
}

// Input data shared in subgroup
// subgroup:   SIMDx8x8 = 8x8x1=64 points
// M=1024, N=1024, K=2560
// 0.222842 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_2(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N / SIMD);
    uint sg_block_b = sg_id / (N / SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;
    //print_wi_info(out_b, out_f);
    int8 result = 0;
    int8 c={};
    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int8 a, b;

        unroll_for(uint bi = 0;bi < BLOCK_B; bi++) {
            a[bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }

        // uint weights_offset = out_f * K + sglid * K + ki;
        // unroll_for(uint fi = 0; fi < BLOCK_F; fi++) {
        //     b[fi]=*(const __global int*)(weights + weights_offset + fi * 4 );
        // }
        b=*(const __global int8*)(weights + out_f * K + sglid * K + ki );

        // print_input_int8(a, sglid);
        // print_input_int8(b, sglid);
        // 0.1 ms
        result += intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
    }

    // %10 - 0.04 ms
    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        // output[output_offset + sglid] = result[bi];
        intel_sub_group_block_write(output + output_offset, result[bi]);
        output_offset += N;
    }
}

// M=1024, N=1024, K=2560
// input + weights data shared in subgroup
// subgroup:   SIMDx8x8 = 8x8x1=64 points
// 0.27715 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_3(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N / SIMD);
    uint sg_block_b = sg_id / (N / SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;
    int8 a = {}, b = {};
    int8 wei;

    // print_wi_info(out_b, out_f);
    int8 result = 0;
    int8 c={};

    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;

        // 0.18 ms
        uint weights_offset = out_f * K + ki;
        unroll_for(uint fi = 0; fi < BLOCK_F; fi++) {
            wei[fi] = as_int(intel_sub_group_block_read((const __global uint*)(weights + weights_offset)));
            weights_offset += K;
        }

        // 0.08 ms
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            a[bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }

#if 0
        int8 data[8];
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            // b[bi] = intel_sub_group_shuffle(wei[0], bi);
            // b[bi] = sub_group_broadcast(wei[sglid], bi);
            unroll_for(uint i = 0; i < BLOCK_F; i ++) {
                data[i][bi] = intel_sub_group_shuffle(wei[i], bi);
            }
        }
        b = data[sglid];
#else
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            // b[bi] = intel_sub_group_shuffle(wei[sglid], bi);
            b[bi] = sub_group_broadcast(wei[sglid], bi);
        }
#endif
        // print_weight_int8(wei, sglid);
        // print_input_int8(b, sglid);
        // 0.1 ms
        result += intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
    }

    // %10 - 0.04 ms
    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        // output[output_offset + sglid] = result[bi];
        intel_sub_group_block_write(output + output_offset, result[bi]);
        output_offset += N;
    }
}

// Input data shared in subgroup, weight data shared in 8 x M 
// subgroup: 4x8xSIMD = 4x8x8=256 points
// lws = 8x1x1
// M=1024, N=1024, K=2560
//  0.162113 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_4(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    // number of 8 x 32, each one employ one dpas
    // uint block_num = M / gws[0];
    #define BLOCK_NUM 4

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N / SIMD);
    uint sg_block_b = sg_id / (N / SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B * BLOCK_NUM;
    //print_wi_info(out_b, out_f);

    int8 result[BLOCK_NUM] = {0};
    int8 c={};
    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int8 a[BLOCK_NUM], b;

        unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
            unroll_for(uint bi = 0;bi < BLOCK_B; bi++) {
                a[num][bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
                input_offset += K;
            }
        }

        // uint weights_offset = out_f * K + sglid * K + ki;
        // unroll_for(uint fi = 0; fi < BLOCK_F; fi++) {
        //     b[fi]=*(const __global int*)(weights + weights_offset + fi * 4 );
        // }
        b=*(const __global int8*)(weights + out_f * K + sglid * K + ki );

        // print_input_int8(a, sglid);
        // print_input_int8(b, sglid);
        unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
            result[num] += intel_sub_group_i8_i8_matrix_mad_k32(a[num],b,c);
        }
    }

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            result[num][bi] += biases[bias_offset + sglid];
            bias_offset += N;
        }
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            intel_sub_group_block_write(output + output_offset, result[num][bi]);
            output_offset += N;
        }
    }
}


// input data shared in subgroup
// subgroup:   SIMDx8x8 = 8x8x1=64 points
// weigh SLM shared in workgroup: 2 subgroup
// M=1024, N=1024, K=2560
//  0.255704 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_5(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();
    uint subgroup_id = get_sub_group_id();

    #define SUBGROUP_TILE_B 2

    uint sg_id = gid/SIMD;
    uint sg_block_f = (sg_id / SUBGROUP_TILE_B) % (N / SIMD);
    uint sg_block_b = (sg_id / (N / SIMD) / SUBGROUP_TILE_B ) * SUBGROUP_TILE_B + sg_id % SUBGROUP_TILE_B;
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;
    // print_wi_info(out_b, out_f);
    int8 result = 0;
    int8 c={};

    __local int8 wei_slm[8];

    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int8 a;
        int8 b;

        unroll_for(uint bi = 0;bi < BLOCK_B; bi++) {
            a[bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }
        if(subgroup_id == 0)
            wei_slm[sglid] = *(const __global int8*)(weights + out_f * K + sglid * K + ki );

        barrier(CLK_LOCAL_MEM_FENCE);
        // b=*(const __global int8*)(weights + out_f * K + sglid * K + ki );
        b = wei_slm[sglid];

        // uint wei_offset = out_f * K + sglid * K + ki;
        // print_input_int4(a, sglid);
        // print_input_int8(b, sglid);
        // 0.1 ms
        result += intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
    }

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        // output[output_offset + sglid] = result[bi];
        intel_sub_group_block_write(output + output_offset, result[bi]);
        output_offset += N;
    }
}


// input data shared across subgroup
// subgroup:   SIMDx8x8 = 8x8x1=64 points
// dpasw
// M=1024, N=1024, K=2560
// 0.295393 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_6(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    uint sg_id = gid/SIMD;
    uint sg_block_f = (sg_id / 2) % (N / SIMD);
    uint sg_block_b = (sg_id / (N / SIMD) / 2 ) * 2 + sg_id % 2;
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B / 2;
    // print_wi_info(out_b, out_f);
    int8 result = 0;
    int8 c={};
    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int4 a;
        int8 b;

        unroll_for(uint bi = 0;bi < BLOCK_B / 2; bi++) {
            a[bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }

        uint weights_offset = out_f * K + sglid * K + ki;
        unroll_for(uint fi = 0; fi < BLOCK_F; fi++) {
            b[fi]=*(const __global int*)(weights + weights_offset + fi * 4 );
        }
        // b=*(const __global int8*)(weights + out_f * K + sglid * K + ki );

        // print_input_int4(a, sglid);
        // print_input_int8(b, sglid);
        // 0.1 ms
        result += intel_sub_group_i8_i8_split_matrix_mad_k32(a,b,c);
    }

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        // output[output_offset + sglid] = result[bi];
        intel_sub_group_block_write(output + output_offset, result[bi]);
        output_offset += N;
    }
}


// M=1024, N=1024, K=2560
// do nothing except XMX + accumulator
// 0.0645594 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_peak(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint SIMD = (uint)get_sub_group_size();
    uint sg_id = gid/SIMD;
    uint out_f = (sg_id % (N / SIMD)) * BLOCK_F;
    uint out_b = (sg_id / (N / SIMD)) * BLOCK_B;

    // print_wi_info(out_b, out_f);
    int8 result = 0;
    int8 a,b,c={};

#if 0
    // 0.0891789 ms
    unroll_for(uint it = 0; it < K; it += BLOCK_K) {
        result += intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
    }
#else
    int8 temp[8] = {};
    uint iter = (K/BLOCK_K/8)*8;
    // 0.0660741 ms
    unroll_for(uint it = 0; it < iter; it += 8) {
        unroll_for(uint i = 0; i < 8; i++){
            temp[i] = intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
        }
        unroll_for(uint i = 0; i < 8; i++){
            result += temp[i];
        }
    }
#endif

    // 0.02 ms
    // uint output_offset = out_b * N + out_f;
    // unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
    //     intel_sub_group_block_write(output + output_offset, result[bi]);
    //     output_offset += N;
    // }
}