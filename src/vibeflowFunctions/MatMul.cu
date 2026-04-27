#include "MatMul.cuh"


namespace VB
{
    __host__ void GEMM(const float* A, 
                       const float* B,
                       float* C,
                       int M, int N, int K)
    {
        using GEMM_CONFIG;

        dim3 block(BN / TN, BM / TM); // (16,16)
        dim3 grid(
            (N + BN - 1) / BN,
            (M + BM - 1) / BM
        );

        gemm_optimized<<<grid, block>>>(A, B, C, M, N, K);
    }

    __global__ void krGEMM(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K)
    {
        using GEMM_CONFIG;

        __shared__ float As[BM][BK];
        __shared__ float Bs[BK][BN];

        const int tx = threadIdx.x;   
        const int ty = threadIdx.y;   

        const int threadRow = ty * TM;
        const int threadCol = tx * TN;

        const int blockRow = blockIdx.y * BM;
        const int blockCol = blockIdx.x * BN;

        float acc[TM][TN];

        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++)
                acc[i][j] = 0.f;
        
        for (int k0 = 0; k0 < K; k0 += BK)
        {
            //
            // Load A tile to shared memory
            //
            for (int i = ty; i < BM; i += blockDim.y)
            {
                for (int j = tx; j < BK; j += blockDim.x)
                {
                    int globalRow = blockRow + i;
                    int globalCol = k0 + j;

                    As[i][j] =
                        (globalRow < M && globalCol < K)
                        ? A[globalRow * K + globalCol]
                        : 0.f;
                }
            }

            //
            // Load B tile to shared memory
            //
            for (int i = ty; i < BK; i += blockDim.y)
            {
                for (int j = tx; j < BN; j += blockDim.x)
                {
                    int globalRow = k0 + i;
                    int globalCol = blockCol + j;

                    Bs[i][j] =
                        (globalRow < K && globalCol < N)
                        ? B[globalRow * N + globalCol]
                        : 0.f;
                }
            }

            __syncthreads();

            //
            // Compute on shared tiles
            //
            #pragma unroll
            for (int k = 0; k < BK; k++)
            {
                float aFrag[TM];
                float bFrag[TN];

                // load A fragment to registers
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    aFrag[i] = As[threadRow + i][k];

                // load B fragment to registers
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    bFrag[j] = Bs[k][threadCol + j];

                // outer product update
                #pragma unroll
                for (int i = 0; i < TM; i++)
                {
                    #pragma unroll
                    for (int j = 0; j < TN; j++)
                    {
                        acc[i][j] += aFrag[i] * bFrag[j];
                    }
                }
            }

            __syncthreads();
        }

        //
        // Store results
        //
        #pragma unroll
        for (int i = 0; i < TM; i++)
        {
            int globalRow = blockRow + threadRow + i;

            if (globalRow < M)
            {
                #pragma unroll
                for (int j = 0; j < TN; j++)
                {
                    int globalCol = blockCol + threadCol + j;

                    if (globalCol < N)
                        C[globalRow * N + globalCol] = acc[i][j];
                }
            }
        }
    }
}
