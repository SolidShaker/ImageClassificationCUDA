#include "MatMul.cuh"


namespace vb
{
    __host__ void gemm(const float* a, 
                       const float* b,
                       float* c,
                       int m, int n, int k)
    {
        using namespace gemm_config;

        dim3 block(bn, bm);   // 128 x 128 threads
        dim3 grid(
            (n + bn - 1) / bn,
            (m + bm - 1) / bm
        );

        krgemm<<<grid, block>>>(a, b, c, m, n, k);
    }
    __global__ void krGEMM(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K)
    {
        using namespace GEMM_CONFIG;

        int row = blockIdx.y * BM + threadIdx.y;
        int col = blockIdx.x * BN + threadIdx.x;

        __shared__ float As[BM][BK];
        __shared__ float Bs[BK][BN];

        float acc = 0.0f;

        for (int k0 = 0; k0 < K; k0 += BK)
        {
            // -------------------------
            // Load A tile
            // -------------------------
            for (int i = threadIdx.y; i < BM; i += blockDim.y)
            {
                for (int j = threadIdx.x; j < BK; j += blockDim.x)
                {
                    int gr = blockIdx.y * BM + i;
                    int gc = k0 + j;

                    As[i][j] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
                }
            }

            // -------------------------
            // Load B tile
            // -------------------------
            for (int i = threadIdx.y; i < BK; i += blockDim.y)
            {
                for (int j = threadIdx.x; j < BN; j += blockDim.x)
                {
                    int gr = k0 + i;
                    int gc = blockIdx.x * BN + j;

                    Bs[i][j] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
                }
            }

            __syncthreads();

            // -------------------------
            // Compute
            // -------------------------
            if (row < M && col < N)
            {
                for (int k = 0; k < BK; ++k)
                {
                    acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }

            __syncthreads();
        }

        // -------------------------
        // Store
        // -------------------------
        if (row < M && col < N)
        {
            C[row * N + col] = acc;
        }
    }
}
