#ifndef MATMUL_CUH
#define MATMUL_CUH

#include "Utils.cuh"
#include <cuda_runtime.h>



namespace GEMM_CONFIG
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;

    constexpr int WARPS_X = 4;
    constexpr int WARPS_Y = 4;

    constexpr int WARP_M = BM / WARPS_Y; // 32
    constexpr int WARP_N = BN / WARPS_X; // 32
}

namespace Vb
{
    __host__ void GEMM(const float* A, 
                       const float* B,
                       float* C,
                       int M, int N, int K);
                     
    __global__ void krGEMM(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K);
}

#endif
