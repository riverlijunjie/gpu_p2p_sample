// #include "../common.cl"
// #include "../sub_group_block_read.cl"
// #include "../sub_group_block_write.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for

#define BLOCK_K 32
#define BLOCK_B 8
#define BLOCK_F 8

// void print_output_int8(int8 result, uint sglid, uint it, uint sg_id) {
//     printf("output[%d:%d]-%d = [%d,%d,%d,%d,%d,%d,%d, %d]\n", sg_id, sglid, it, result[0], result[1],result[2],result[3],result[4],result[5],result[6],result[7]);
// }

void print_output_int8(int8 result, int m, int n) {
    uint sglid = (uint)get_sub_group_local_id();
    uint sg_c = get_local_id(0);
    uint sg_n = get_local_id(1);
    uint sg_m = get_local_id(2);
    uint sg_id = get_sub_group_id(); // sg_m * WG_SG_NUM_N + sg_n; 
    printf("output[%d][%d]-%d = [%d,%d,%d,%d,%d,%d,%d, %d]\n", m, n, sg_id, result[0], result[1],result[2],result[3],result[4],result[5],result[6],result[7]);
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

void print_weight_int8(int8 result, uint sglid) {
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

// M=1024, N=1024, K=2560
// Raw version, without memory share in subgroup
// 7 TOPS/s
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
// 25 TOPS/s
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

    int8 a,b, result;
    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
        result[bi] = biases[bias_offset + sglid];
        bias_offset += N;
    }

    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
            a[bi] = *(__global int *)(input + input_offset + sglid * 4);
            input_offset += K;
        }
        uint weights_offset = out_f * K + sglid * K + ki;
        // unroll_for(uint fi =0; fi < BLOCK_F; fi++) {
        //     b[fi] = *(__global int *)(weights + weights_offset + fi * 4);
        // }
        b=*(const __global int8*)(weights + weights_offset );
        result = intel_sub_group_i8_i8_matrix_mad_k32(a,b,result);
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
    int8 result;

    // %10 - 0.04 ms
    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] = biases[bias_offset + sglid];
        bias_offset += N;
    }

    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int8 a, b;

        unroll_for(uint bi = 0;bi < BLOCK_B; bi++) {
            a[bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }
        b=*(const __global int8*)(weights + out_f * K + sglid * K + ki );
        result = intel_sub_group_i8_i8_matrix_mad_k32(a,b,result);
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        // output[output_offset + sglid] = result[bi];
        intel_sub_group_block_write(output + output_offset, result[bi]);
        output_offset += N;
    }
}

// M=1024, N=1024, K=2560
// input/weight block read + weights data shared in subgroup
// subgroup:   SIMDx8x8 = 8x8x1=64 points
// 22.69 TOPS/s
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
    int8 a,b,wei, result;

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] = biases[bias_offset + sglid];
        bias_offset += N;
    }

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
            unroll_for(uint i = 0; i < BLOCK_F; i ++) {
                data[i][bi] = intel_sub_group_shuffle(wei[i], bi);
            }
        }
        b = data[sglid];
#else
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            b[bi] = intel_sub_group_shuffle(wei[bi], sglid);
        }
#endif
        result = intel_sub_group_i8_i8_matrix_mad_k32(a,b,result);
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        // output[output_offset + sglid] = result[bi];
        intel_sub_group_block_write(output + output_offset, result[bi]);
        output_offset += N;
    }
}

// 4 XMX in each subgroup
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

    int8 result[BLOCK_NUM];
    uint bias_offset = out_b * N + out_f;
    unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            result[num][bi] = biases[bias_offset + sglid];
            bias_offset += N;
        }
    }

    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int8 a[BLOCK_NUM], b;

        unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
            unroll_for(uint bi = 0;bi < BLOCK_B; bi++) {
                // a[num][bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
                a[num][bi] = *(__global int *)(input + input_offset + sglid * 4);
                input_offset += K;
            }
        }
        b=*(const __global int8*)(weights + out_f * K + sglid * K + ki );

        //__attribute__((opencl_unroll_hint(BLOCK_NUM)))
        unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
            result[num] = intel_sub_group_i8_i8_matrix_mad_k32(a[num],b,result[num]);
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


// subgroup: 8x4 XMX
// input/weight SLM shared in workgroup
// M=1024, N=1024, K=2560
#define dK 32   // int8 is 32
#define dX 4    // How many iter in BK
#define BK (dK*dX)  // 128
#define WG_SG_NUM_M 4
#define WG_SG_NUM_N 8
#define SG_BLOCK_NUM_M 4
#define SG_BLOCK_NUM_N 2

#define SG_OUTPUT_M (SG_BLOCK_NUM_M * 8)   //32
#define SG_OUTPUT_N (SG_BLOCK_NUM_N * 8)   //16

#define WG_OUTPUT_M (WG_SG_NUM_M * SG_OUTPUT_M)   // 4*32=128
#define WG_OUTPUT_N (WG_SG_NUM_N * SG_OUTPUT_N)   // 8*16=128

#define SG_INPUT_BLOCK_SIZE (SG_OUTPUT_M * BK)   // 32*128=4096
#define SG_WEIGHT_BLOCK_SIZE (SG_OUTPUT_N * BK)  // 16*128=2048

#define XMX_INPUT_BLOCK_SIZE (8 * dK)  //  8*32=256

#define SG_COPY_INPUT_BLOCK_SIZE (SG_OUTPUT_M * BK / WG_SG_NUM_N)   // 32*128/8 = 512
#define SG_COPY_INPUT_BLOCK_LINE (SG_COPY_INPUT_BLOCK_SIZE / 32)   // 512/32=16
#define WI_COPY_INPUT_BLOCK_SIZE (SG_COPY_INPUT_BLOCK_SIZE / 8)    // 512/8=64
#define WI_COPY_INPUT_BLOCK_LINE (WI_COPY_INPUT_BLOCK_SIZE / 32)   // 64/32=2

#define SG_COPY_WEIGHT_BLOCK_SIZE (SG_OUTPUT_N * BK / WG_SG_NUM_M)  // 16*128/4 = 512
#define SG_COPY_WEIGHT_BLOCK_LINE (SG_COPY_WEIGHT_BLOCK_SIZE / 32)  // 512/32=16
#define WI_COPY_WEIGHT_BLOCK_SIZE (SG_COPY_WEIGHT_BLOCK_SIZE / 8)    // 512/8=64
#define WI_COPY_WEIGHT_BLOCK_LINE (WI_COPY_WEIGHT_BLOCK_SIZE / 32)   // 64/32=2

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_5(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint SIMD = (uint)get_sub_group_size();

    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
   
    uint sglid = (uint)get_sub_group_local_id();
    uint sg_c = get_local_id(0);
    uint sg_n = get_local_id(1);
    uint sg_m = get_local_id(2);
    uint sg_id = get_sub_group_id(); // sg_m * WG_SG_NUM_N + sg_n; 
    
    __local char in_slm[WG_OUTPUT_M * BK]; // 128*128/1024=16KB
    __local char wei_slm[WG_OUTPUT_N * BK];// 128*128/1024=16KB

    #if WI_COPY_INPUT_BLOCK_LINE==0 || WI_COPY_WEIGHT_BLOCK_LINE==0
    #error "WI_COPY_INPUT_BLOCK_LINE or WI_COPY_WEIGHT_BLOCK_LINE is 0"
    #endif

    int8 result[SG_BLOCK_NUM_M][SG_BLOCK_NUM_N];
    {
        uint bias_offset = (wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M) * N + (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N);
        unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
            unroll_for(uint n = 0; n < SG_BLOCK_NUM_N; n++) {
                unroll_for(uint c = 0; c < 8; c++)
                    result[m][n][c] = (int)(biases[bias_offset + n * 8 + sg_c + c * N]);
            }
            bias_offset += 8 * N;
        }
    }

    unroll_for(uint ki = 0; ki < K; ki += BK) {
        // Copy input data into local memory
        {
            // uint local_input_offset = (sg_m * WG_SG_NUM_N + sg_n) * SG_COPY_INPUT_BLOCK_SIZE + sg_c * WI_COPY_INPUT_BLOCK_SIZE;
            uint local_input_index = sg_m * SG_BLOCK_NUM_M * dX + sg_n * WI_COPY_INPUT_BLOCK_LINE;
            uint local_input_offset = local_input_index * XMX_INPUT_BLOCK_SIZE + sg_c * dK; 
            // uint m = wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M + (sg_n * SG_COPY_INPUT_BLOCK_LINE) % SG_OUTPUT_M + sg_c * WI_COPY_INPUT_BLOCK_LINE;
            uint m = wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M + (sg_n * WI_COPY_INPUT_BLOCK_LINE / dX) * 8 + sg_c;
            uint input_offset = m * K + ki + (sg_n * WI_COPY_INPUT_BLOCK_LINE * dK) % BK;
            unroll_for(uint i = 0; i < WI_COPY_INPUT_BLOCK_LINE; i++) {
                *(__local int8 *)(in_slm + local_input_offset + i * XMX_INPUT_BLOCK_SIZE) = *(__global int8 *)(input + input_offset + i * dK);
            }

            //if(sg_c==0)
            //{
            //    printf("input: wg = (%d,%d), sg = (%d,%d), sg_id = %d, sg_c = %d, local_index = %d, input = (%d, %d)\n",
            //           wg_m, wg_n, sg_m, sg_n, sg_id, sg_c, local_input_index, m, input_offset % K);
            //}

        }
        // Copy weight data into local memory
        {
            // uint local_wei_offset = (sg_m * WG_SG_NUM_N + sg_n) * SG_COPY_WEIGHT_BLOCK_SIZE; // block_write
            uint local_wei_index = sg_n * SG_BLOCK_NUM_N * dX + sg_m * WI_COPY_WEIGHT_BLOCK_LINE;
            uint local_wei_offset = local_wei_index * XMX_INPUT_BLOCK_SIZE;// + sg_c * dK; // block_write
            //  uint n = wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N + (sg_m * SG_COPY_WEIGHT_BLOCK_LINE) % WG_SG_NUM_N + sg_c * WI_COPY_WEIGHT_BLOCK_LINE;
            uint n = wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE / dX) * 8 + sg_c;
            uint wei_offset =  n * K + ki + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE * dK) % BK;
            unroll_for(uint i = 0; i < WI_COPY_WEIGHT_BLOCK_LINE; i++) {
                int8 wei_cols_data =  *(__global int8 *)(weights + wei_offset + i * dK);
                intel_sub_group_block_write8((__local uint*)(wei_slm + local_wei_offset + i * XMX_INPUT_BLOCK_SIZE), as_uint8(wei_cols_data));

            }
            //unroll_for(uint i = 0; i < WI_COPY_WEIGHT_BLOCK_LINE; i++) {
            //    *(__local int8 *)(wei_slm + local_wei_offset + i * XMX_INPUT_BLOCK_SIZE) = *(__global int8 *)(weights + wei_offset + i * dK);
            //}
            //if(sg_c==0)
            //{
            //    printf("weight: wg = (%d,%d), sg = (%d,%d), sg_id = %d, sg_c = %d, local_index = %d, input = (%d, %d)\n",
            //           wg_m, wg_n, sg_m, sg_n, sg_id, sg_c, local_wei_index, n, wei_offset % K);
            //}
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        const __local char* local_input_ptr = (const __local char *)(in_slm + sg_m * SG_INPUT_BLOCK_SIZE);
        const __local char* local_wei_ptr  = (const __local char *)(wei_slm  + sg_n * SG_WEIGHT_BLOCK_SIZE);
        __attribute__((opencl_unroll_hint(1)))
        for(int kii = 0; kii < dX; kii++) {
            int8 in_vec[SG_BLOCK_NUM_M];
            int8 wei_vec[SG_BLOCK_NUM_N];
            
            unroll_for(int m = 0; m < SG_BLOCK_NUM_M; m++) {
                in_vec[m] = as_int8(intel_sub_group_block_read8((__local uint*)(local_input_ptr + (kii + m * dX)* XMX_INPUT_BLOCK_SIZE)));
            }

            // if(sg_m==3 && wg_m==1)
            // {
            //      print_input_int8(in_vec[3], sg_n);
            // }
    
            unroll_for(int n = 0; n < SG_BLOCK_NUM_N; n++) {
                wei_vec[n] = as_int8(intel_sub_group_block_read8((__local uint*)(local_wei_ptr + (kii + n * dX)* XMX_INPUT_BLOCK_SIZE)));
            }

            // if(sg_m==3 && wg_m==1)
            // {
            //    print_weight_int8(wei_vec[1], kii);
            // }

            unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
                unroll_for(uint n =0; n < SG_BLOCK_NUM_N; n++) {
                    result[m][n] = intel_sub_group_i8_i8_matrix_mad_k32(in_vec[m], wei_vec[n], result[m][n]);
                    // if(sg_m==3 && wg_m==1)
                    //    print_output_int8(result[m][n], m, n);
                }
            }
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // __local char* scratch = in_slm + sg_id * SG_INPUT_BLOCK_SIZE * SG_BLOCK_NUM_M * SG_BLOCK_NUM_N;
    // unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
    //     unroll_for(uint n = 0; n < SG_BLOCK_NUM_N; n++) {
    //         intel_sub_group_block_write8((__local uint *)(scratch + (m * SG_BLOCK_NUM_N + n) * SG_INPUT_BLOCK_SIZE), as_uint8(result[m][n]));
    //     }
    // }

    __global int *dst_ptr = output + (wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M) * N + (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N) + sg_c;
    unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
        unroll_for(uint n = 0; n < SG_BLOCK_NUM_N; n++) {
            unroll_for(uint c = 0; c < 8; c++) {
                dst_ptr[c * N + n * 8] = result[m][n][c];
            }
        }
        dst_ptr += 8 * N;
    }
}

// dpasw
// subgroup: 8x4 XMX
// input/weight SLM shared in workgroup
// M=1024, N=1024, K=2560
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_5_dpasw(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint SIMD = (uint)get_sub_group_size();

    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
   
    uint sglid = (uint)get_sub_group_local_id();
    uint sg_c = get_local_id(0);
    uint sg_n = get_local_id(1);
    uint sg_m = get_local_id(2);
    uint sg_id = get_sub_group_id(); // sg_m * WG_SG_NUM_N + sg_n; 
    
    __local char in_slm[WG_OUTPUT_M * BK]; // 128*128/1024=16KB
    __local char wei_slm[WG_OUTPUT_N * BK];// 128*128/1024=16KB

    #if WI_COPY_INPUT_BLOCK_LINE==0 || WI_COPY_WEIGHT_BLOCK_LINE==0
    #error "WI_COPY_INPUT_BLOCK_LINE or WI_COPY_WEIGHT_BLOCK_LINE is 0"
    #endif

    int8 result[SG_BLOCK_NUM_M][SG_BLOCK_NUM_N];
    {
        uint bias_offset = (wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M) * N + (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N);
        unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
            unroll_for(uint n = 0; n < SG_BLOCK_NUM_N; n++) {
                unroll_for(uint c = 0; c < 8; c++)
                    result[m][n][c] = (int)(biases[bias_offset + n * 8 + sg_c + c * N]);
            }
            bias_offset += 8 * N;
        }
    }

    unroll_for(uint ki = 0; ki < K; ki += BK) {
        // Copy input data into local memory
        {
            uint local_input_index = sg_m * SG_BLOCK_NUM_M * dX + sg_n * WI_COPY_INPUT_BLOCK_LINE;
            uint local_input_offset = local_input_index * XMX_INPUT_BLOCK_SIZE + sg_c * dK; 
            uint m = wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M + (sg_n * WI_COPY_INPUT_BLOCK_LINE / dX) * 8 + sg_c;
            uint input_offset = m * K + ki + (sg_n * WI_COPY_INPUT_BLOCK_LINE * dK) % BK;
            unroll_for(uint i = 0; i < WI_COPY_INPUT_BLOCK_LINE; i++) {
                *(__local int8 *)(in_slm + local_input_offset + i * XMX_INPUT_BLOCK_SIZE) = *(__global int8 *)(input + input_offset + i * dK);
            }
        }
        // Copy weight data into local memory
        {
            uint local_wei_index = sg_n * SG_BLOCK_NUM_N * dX + sg_m * WI_COPY_WEIGHT_BLOCK_LINE;
            uint local_wei_offset = local_wei_index * XMX_INPUT_BLOCK_SIZE;// + sg_c * dK; // block_write
            uint n = wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE / dX) * 8 + sg_c;
            uint wei_offset =  n * K + ki + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE * dK) % BK;
            unroll_for(uint i = 0; i < WI_COPY_WEIGHT_BLOCK_LINE; i++) {
                int8 wei_cols_data =  *(__global int8 *)(weights + wei_offset + i * dK);
                intel_sub_group_block_write8((__local uint*)(wei_slm + local_wei_offset + i * XMX_INPUT_BLOCK_SIZE), as_uint8(wei_cols_data));

            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        const __local char* local_input_ptr = (const __local char *)(in_slm + sg_m * SG_INPUT_BLOCK_SIZE);
        const __local char* local_wei_ptr  = (const __local char *)(wei_slm  + sg_n * SG_WEIGHT_BLOCK_SIZE);
        __attribute__((opencl_unroll_hint(1)))
        for(int kii = 0; kii < dX; kii++) {
            int4 in_vec[SG_BLOCK_NUM_M];
            int8 wei_vec[SG_BLOCK_NUM_N];
            
            unroll_for(int m = 0; m < SG_BLOCK_NUM_M; m++) {
                in_vec[m] = as_int4(intel_sub_group_block_read4((__local uint*)(local_input_ptr + (kii + m * dX)* XMX_INPUT_BLOCK_SIZE + (sg_id % 2)* (4*32))));
            }

            unroll_for(int n = 0; n < SG_BLOCK_NUM_N; n++) {
                wei_vec[n] = as_int8(intel_sub_group_block_read8((__local uint*)(local_wei_ptr + (kii + n * dX)* XMX_INPUT_BLOCK_SIZE)));
            }

            unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
                unroll_for(uint n =0; n < SG_BLOCK_NUM_N; n++) {
                    result[m][n] = intel_sub_group_i8_i8_split_matrix_mad_k32(in_vec[m], wei_vec[n], result[m][n]);
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
            unroll_for(uint c = 0; c < 8; c++) {
                dst_ptr[c * N + n * 8] = result[m][n][c];
            }
        }
        dst_ptr += 8 * N;
    }
}

// weight repack in tile
// dpasw
// subgroup: 8x4 XMX
// input/weight SLM shared in workgroup
// M=1024, N=1024, K=2560
//
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_6_dpasw_repack(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint SIMD = (uint)get_sub_group_size();

    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
   
    uint sglid = (uint)get_sub_group_local_id();
    uint sg_c = get_local_id(0);
    uint sg_n = get_local_id(1);
    uint sg_m = get_local_id(2);
    uint sg_id = get_sub_group_id(); // sg_m * WG_SG_NUM_N + sg_n; 
    
    __local char in_slm[WG_OUTPUT_M * BK]; // 128*128/1024=16KB
    __local char wei_slm[WG_OUTPUT_N * BK];// 128*128/1024=16KB

    #if WI_COPY_INPUT_BLOCK_LINE==0 || WI_COPY_WEIGHT_BLOCK_LINE==0
    #error "WI_COPY_INPUT_BLOCK_LINE or WI_COPY_WEIGHT_BLOCK_LINE is 0"
    #endif

    int8 result[SG_BLOCK_NUM_M][SG_BLOCK_NUM_N];
    {
        uint bias_offset = (wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M) * N + (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N);
        unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
            unroll_for(uint n = 0; n < SG_BLOCK_NUM_N; n++) {
                unroll_for(uint c = 0; c < 8; c++)
                    result[m][n][c] = (int)(biases[bias_offset + n * 8 + sg_c + c * N]);
            }
            bias_offset += 8 * N;
        }
    }

    unroll_for(uint ki = 0; ki < K; ki += BK) {
        #if 1
        // repack - Copy input data into local memory
        {
            uint local_input_index = sg_m * SG_BLOCK_NUM_M * dX + sg_n * WI_COPY_INPUT_BLOCK_LINE;
            uint local_input_offset = local_input_index * XMX_INPUT_BLOCK_SIZE; 
            uint m = wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M;
            uint input_offset = m * K + ki + local_input_offset;
            unroll_for(uint i = 0; i < WI_COPY_INPUT_BLOCK_LINE; i++) {
                uint8 value = intel_sub_group_block_read8((__global uint*)(input + input_offset + i * XMX_INPUT_BLOCK_SIZE));
                intel_sub_group_block_write8((__local uint*)(wei_slm + local_input_offset + i * XMX_INPUT_BLOCK_SIZE), value);
            }
        }
        #else
        // no repack - Copy input data into local memory
        {
            uint local_input_index = sg_m * SG_BLOCK_NUM_M * dX + sg_n * WI_COPY_INPUT_BLOCK_LINE;
            uint local_input_offset = local_input_index * XMX_INPUT_BLOCK_SIZE + sg_c * dK; 
            uint m = wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M + (sg_n * WI_COPY_INPUT_BLOCK_LINE / dX) * 8 + sg_c;
            uint input_offset = m * K + ki + (sg_n * WI_COPY_INPUT_BLOCK_LINE * dK) % BK;
            unroll_for(uint i = 0; i < WI_COPY_INPUT_BLOCK_LINE; i++) {
                *(__local int8 *)(in_slm + local_input_offset + i * XMX_INPUT_BLOCK_SIZE) = *(__global int8 *)(input + input_offset + i * dK);
            }
        }
        #endif


        #if 1
        // repack - Copy weight data into local memory
        {
            uint local_wei_index = sg_n * SG_BLOCK_NUM_N * dX + sg_m * WI_COPY_WEIGHT_BLOCK_LINE;
            uint local_wei_offset = local_wei_index * XMX_INPUT_BLOCK_SIZE;// + sg_c * dK; // block_write
            uint n = wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N;
            uint wei_offset =  n * K + ki + local_wei_offset;
            unroll_for(uint i = 0; i < WI_COPY_WEIGHT_BLOCK_LINE; i++) {
                uint8 value = intel_sub_group_block_read8((__global uint*)(weights + local_wei_offset + i * XMX_INPUT_BLOCK_SIZE));
                intel_sub_group_block_write8((__local uint*)(wei_slm + local_wei_offset + i * XMX_INPUT_BLOCK_SIZE), value);
            }
        }
        #else
        // no repack - Copy weight data into local memory
        {
            uint local_wei_index = sg_n * SG_BLOCK_NUM_N * dX + sg_m * WI_COPY_WEIGHT_BLOCK_LINE;
            uint local_wei_offset = local_wei_index * XMX_INPUT_BLOCK_SIZE;// + sg_c * dK; // block_write
            uint n = wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE / dX) * 8 + sg_c;
            uint wei_offset =  n * K + ki + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE * dK) % BK;
            unroll_for(uint i = 0; i < WI_COPY_WEIGHT_BLOCK_LINE; i++) {
                int8 wei_cols_data =  *(__global int8 *)(weights + wei_offset + i * dK);
                intel_sub_group_block_write8((__local uint*)(wei_slm + local_wei_offset + i * XMX_INPUT_BLOCK_SIZE), as_uint8(wei_cols_data));

            }
        }
        #endif

        barrier(CLK_LOCAL_MEM_FENCE);
        const __local char* local_input_ptr = (const __local char *)(in_slm + sg_m * SG_INPUT_BLOCK_SIZE);
        const __local char* local_wei_ptr  = (const __local char *)(wei_slm  + sg_n * SG_WEIGHT_BLOCK_SIZE);
        __attribute__((opencl_unroll_hint(1)))
        for(int kii = 0; kii < dX; kii++) {
            int4 in_vec[SG_BLOCK_NUM_M];
            int8 wei_vec[SG_BLOCK_NUM_N];
            
            unroll_for(int m = 0; m < SG_BLOCK_NUM_M; m++) {
                in_vec[m] = as_int4(intel_sub_group_block_read4((__local uint*)(local_input_ptr + (kii + m * dX)* XMX_INPUT_BLOCK_SIZE + (sg_id % 2)* (4*32))));
            }

            unroll_for(int n = 0; n < SG_BLOCK_NUM_N; n++) {
                wei_vec[n] = as_int8(intel_sub_group_block_read8((__local uint*)(local_wei_ptr + (kii + n * dX)* XMX_INPUT_BLOCK_SIZE)));
            }

            unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
                unroll_for(uint n =0; n < SG_BLOCK_NUM_N; n++) {
                    result[m][n] = intel_sub_group_i8_i8_split_matrix_mad_k32(in_vec[m], wei_vec[n], result[m][n]);
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
            unroll_for(uint c = 0; c < 8; c++) {
                dst_ptr[c * N + n * 8] = result[m][n][c];
            }
        }
        dst_ptr += 8 * N;
    }
}


// M=1024, N=1024, K=2560
// do nothing except XMX + accumulator
// 0.0359509 ms
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_peak(__global char* input, __global char* weights, __global char* biases, __global int* output, int M, int N, int K) {

    uint gid = (uint)get_global_id(0);
    uint SIMD = (uint)get_sub_group_size();

    // int wg_n = get_group_id(1);
    // int wg_m = get_group_id(2);
    // uint sglid = (uint)get_sub_group_local_id();
    // uint sg_c = get_local_id(0);
    // uint sg_n = get_local_id(1);
    // uint sg_m = get_local_id(2);
    // uint sg_id = get_sub_group_id(); // sg_m * WG_SG_NUM_N + sg_n; 
    // uint m = wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M + (sg_n * WI_COPY_INPUT_BLOCK_LINE / dX) * 8 + sg_c;
    // uint input_offset = m * K + (sg_n * WI_COPY_INPUT_BLOCK_LINE * dK) % BK;
    // uint n = wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE / dX) * 8 + sg_c;
    // uint wei_offset =  n * K + (sg_m * WI_COPY_WEIGHT_BLOCK_LINE * dK) % BK;
    // int4 a = as_int4(intel_sub_group_block_read4((__global uint*) (input + input_offset)));
    // int8 b = as_int8(intel_sub_group_block_read8((__global uint*) (weights + wei_offset)));

    int4 a = as_int4(intel_sub_group_block_read4((__global uint*) (input)));
    int8 b = as_int8(intel_sub_group_block_read8((__global uint*) (weights)));

    int8 result[SG_BLOCK_NUM_M][SG_BLOCK_NUM_N];
#if 0
    // 116 TOPS/s - 1024 2048 2560
    unroll_for(uint it = 0; it < K; it += BLOCK_K) {
        result = intel_sub_group_i8_i8_matrix_mad_k32(a,b,result);
    }
#else
    unroll_for(uint ki = 0; ki < K; ki += BK) {
        //__attribute__((opencl_unroll_hint(1)))
        unroll_for(int kii = 0; kii < dX; kii++) {
            unroll_for(uint m = 0; m < SG_BLOCK_NUM_M; m++) {
                unroll_for(uint n =0; n < SG_BLOCK_NUM_N; n++) {
                    result[m][n] = intel_sub_group_i8_i8_split_matrix_mad_k32(a, b, result[m][n]);
                }
            }
        }
    }

#endif
    // intel_sub_group_block_write8((__global uint*)output, as_uint8(result));

    // 0.012 ms
    // uint output_offset = out_b * N + out_f;
    // unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
    //     intel_sub_group_block_write(output + output_offset, result[bi]);
    //     output_offset += N;
    // }
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_peak_(__global char * A, __global char * B, __global char * C,  __global char * D, int M, int N, int K) {
    int8 a = as_int8(intel_sub_group_block_read8((__global uint*) A));
    int8 b = as_int8(intel_sub_group_block_read8((__global uint*) B));

    #define ACC_CNT 8
    int8 acc[ACC_CNT];

    __attribute__((opencl_unroll_hint(1)))
    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K * ACC_CNT) {
        __attribute__((opencl_unroll_hint(ACC_CNT)))
        for(int k = 0; k < ACC_CNT; k++)
            acc[k] = intel_sub_group_i8_i8_matrix_mad_k32(a, b, acc[k]);
    }
    //int8 result = acc[0];
    //unroll_for(uint i = 1; i < ACC_CNT; i++){
    //    result += acc[i];
    //}

    intel_sub_group_block_write8((__global uint*)C, as_uint8(acc[0]));
    //barrier(CLK_LOCAL_MEM_FENCE);
}
