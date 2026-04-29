#ifndef TEST_GEMM_CUH
#define TEST_GEMM_CUH

#include <cuda_runtime.h>

namespace Test
{
    __global__ void krGEMM(const float* A,
                           const float* B,
                           float* C,
                           int M, int N, int K)
    {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.x * blockDim.x + threadIdx.x;

        if (r < M && c < N)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; k++)
                sum += A[r * K + k] * B[k * N + c];

            C[r * N + c] = sum;
        }
    }

    __host__ void GEMM(const float* A,
                       const float* B,
                       float* C,
                       int M, int N, int K)
    {
        dim3 block(16, 16);
        dim3 grid(
            (N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y
        );

        krGEMM<<<grid, block>>>(A, B, C, M, N, K);
    }
}

#endif
