#include "MatMul.cuh"


namespace Vb
{
    __host__ void GEMM(const float* A, 
                       const float* B,
                       float* C,
                       int M, int N, int K)
    {
        using namespace GEMM_CONFIG;

        dim3 blockDim(WARPS_X * WARPS_Y * 32);  // 512 threads
        dim3 gridDim(
            (N + BN - 1) / BN,
            (M + BM - 1) / BM
        );

        krGEMM<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }

    __global__ void krGEMM(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K)
    {
        using namespace GEMM_CONFIG;
        int warpId = threadIdx.x / 32;
        int laneId = threadIdx.x % 32;

        int warpRow = warpId / WARPS_X;
        int warpCol = warpId % WARPS_X;

        int rowBase = blockIdx.y * BM + warpRow * WARP_M;
        int colBase = blockIdx.x * BN + warpCol * WARP_N;

        __shared__ float As[BM][BK];
        __shared__ float Bs[BK][BN];

        float acc[2] = {0.0f}; // register fragment (no TM/TN loops)

        for (int k0 = 0; k0 < K; k0 += BK)
        {
            // -------------------------
            // Load A and B tiles
            // -------------------------
            int row = threadIdx.x / BK;
            int col = threadIdx.x % BK;

            if (row < BM && (k0 + col) < K)
                As[row][col] = A[(blockIdx.y * BM + row) * K + (k0 + col)];

            if (col < BN && (k0 + row) < K)
                Bs[row][col] = B[(k0 + row) * N + (blockIdx.x * BN + col)];

            __syncthreads();

            // -------------------------
            // Warp-level computation
            // -------------------------
            for (int k = 0; k < BK; ++k)
            {
                float a = As[rowBase + laneId / WARP_N][k];
                float b = Bs[k][colBase + laneId % WARP_N];

                acc[0] += a * b;
            }

            __syncthreads();
        }

        // -------------------------
        // Store results
        // -------------------------
        int r = rowBase + laneId / WARP_N;
        int c = colBase + laneId % WARP_N;

        if (r < M && c < N)
            C[r * N + c] = acc[0];
    }
}
