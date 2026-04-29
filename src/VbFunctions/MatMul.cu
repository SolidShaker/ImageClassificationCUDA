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

        const int tx = threadIdx.x;
        const int warpId = tx / 32;
        const int laneId = tx % 32;

        const int warpRow = warpId / WARPS_X;
        const int warpCol = warpId % WARPS_X;

        const int baseRow = blockIdx.y * BM + warpRow * WARP_M;
        const int baseCol = blockIdx.x * BN + warpCol * WARP_N;

        __shared__ float As[BM][BK];
        __shared__ float Bs[BK][BN];

        float acc[4] = {0.f, 0.f, 0.f, 0.f}; // small register tile

        // lane mapping inside warp (8x4 = 32 threads)
        const int laneRow = laneId / 8;
        const int laneCol = laneId % 8;

        for (int k0 = 0; k0 < K; k0 += BK)
        {
            // =====================================================
            // LOAD A (coalesced)
            // =====================================================
            for (int i = tx; i < BM * BK; i += blockDim.x)
            {
                int r = i / BK;
                int c = i % BK;

                int gr = blockIdx.y * BM + r;
                int gc = k0 + c;

                As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
            }

            // =====================================================
            // LOAD B (coalesced)
            // =====================================================
            for (int i = tx; i < BK * BN; i += blockDim.x)
            {
                int r = i / BN;
                int c = i % BN;

                int gr = k0 + r;
                int gc = blockIdx.x * BN + c;

                Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
            }

            __syncthreads();

            // =====================================================
            // WARP COMPUTE
            // =====================================================
            for (int k = 0; k < BK; ++k)
            {
                float a = As[baseRow + laneRow][k];
                float b = Bs[k][baseCol + laneCol];

                // outer product (simple but correct)
                acc[0] += a * b;
            }

            __syncthreads();
        }

        // =========================================================
        // STORE
        // =========================================================
        int r = baseRow + laneRow;
        int c = baseCol + laneCol;

        if (r < M && c < N)
            C[r * N + c] = acc[0];    
    }
}
