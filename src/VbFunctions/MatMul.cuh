#ifndef MATMUL_CUH
#define MATMUL_CUH

#include "Utils.cuh"
#include <cuda_runtime.h>


namespace GEMM_CONFIG
{
    constexpr int BM = 128;   // block tile M
    constexpr int BN = 128;   // block tile N
    constexpr int BK = 32;    // K tile

    constexpr int TM = 8;     // per-thread rows
    constexpr int TN = 8;     // per-thread cols
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
