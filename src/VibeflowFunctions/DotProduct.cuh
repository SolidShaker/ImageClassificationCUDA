#ifndef DOTPRODUCT
#define DOTPRODUCT

#include "Utils.cuh"
#include <cuda_runtime.h>



namespace DOT_CONFIG
{
    constexpr int BLOCK_SIZE = 256;
};


namespace VB
{
    __device__ void Dot(const float* A,
                        const float* B,
                        float* result,
                        int N);
    // REQUIRED: ZERO-PADDING
    __global__ void krDotStart(const float* __restrict__ A,     
                               const float* __restrict__ B,
                               float* __restrict__ blockSum,
                               int N);
    __global__ void krDotEnd(const float* __restrict__ blockSum,
                             float* __restrict__ result,
                             int N);
}

#endif
