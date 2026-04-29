#include "MatMul.cuh"


namespace Vb
{

    __host__ void GEMM(const float* A, 
                       const float* B,
                       float* C,
                       int M, int N, int K)
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(
                (N + TILE_SIZE - 1) / TILE_SIZE, 
                (M + TILE_SIZE - 1) / TILE_SIZE
        );
        krGEMM<<<grid, block>>>(A, B, C, M, N, K);
    }
    __global__ void krGEMM(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K)
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = blockIdx.y * TILE_SIZE + ty;
        int col = blockIdx.x * TILE_SIZE + tx;

        __shared__ float4 tileA[TILE_SIZE][TILE_SIZE];
        __shared__ float4 tileB[TILE_SIZE][TILE_SIZE];

        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* B4 = reinterpret_cast<const float4*>(B);

        float sum = 0.0f;

        int K4 = K / 4;

        for (int k0 = 0; k0 < (K4 + TILE_SIZE - 1) / TILE_SIZE; ++k0)
        {
            int colA = k0 * TILE_SIZE + tx;

            tileA[ty][tx] =
                (row < M && colA < K4)
                ? A4[row * K4 + colA]
                : make_float4(0.f, 0.f, 0.f, 0.f);

            int rowB = k0 * TILE_SIZE + ty;

            tileB[ty][tx] =
                (rowB < K4 && col < N)
                ? B4[rowB * N + col]
                : make_float4(0.f, 0.f, 0.f, 0.f);

            __syncthreads();

            for (int k = 0; k < TILE_SIZE; ++k)
            {
                float4 a = tileA[ty][k];
                float4 b = tileB[k][tx];

                sum += a.x * b.x;
                sum += a.y * b.y;
                sum += a.z * b.z;
                sum += a.w * b.w;
            }

            __syncthreads();
        }

        if (row < M && col < N)
        {
            C[row * N + col] = sum;
        }
    }    

    // __host__ void GEMM(const float* A, 
    //                    const float* B,
    //                    float* C,
    //                    int M, int N, int K)
    // {
    //     using namespace GEMM_CONFIG;
    //
    //     dim3 block(BN / TN, BM / TM); // (16,16)
    //     dim3 grid(
    //         (N + BN - 1) / BN,
    //         (M + BM - 1) / BM
    //     );
    //
    //     krGEMM<<<grid, block>>>(A, B, C, M, N, K);
    // }
    //
    // __global__ void krGEMM(const float* __restrict__ A,
    //                        const float* __restrict__ B,
    //                        float* __restrict__ C,
    //                        int M, int N, int K)
    // {
    //     using namespace GEMM_CONFIG;
    //
    //     const int tx = threadIdx.x;   
    //     const int ty = threadIdx.y;   
    //
    //     const int threadRow = ty * TM;
    //     const int threadCol = tx * TN;
    //
    //     const int blockRow = blockIdx.y * BM;
    //     const int blockCol = blockIdx.x * BN;
    //
    //     __shared__ float As[BM][BK];
    //     __shared__ float Bs[BK][BN];
    //
    //     float acc[TM][TN];
    //
    //     #pragma unroll
    //     for (int i = 0; i < TM; i++)
    //         #pragma unroll
    //         for (int j = 0; j < TN; j++)
    //             acc[i][j] = 0.f;
    //
    //     for (int k0 = 0; k0 < K; k0 += BK)
    //     {
    //         //
    //         // Load A tile to shared memory
    //         //
    //         for (int i = ty; i < BM; i += blockDim.y)
    //         {
    //             for (int j = tx; j < BK; j += blockDim.x)
    //             {
    //                 int globalRow = blockRow + i;
    //                 int globalCol = k0 + j;
    //
    //                 As[i][j] =
    //                     (globalRow < M && globalCol < K)
    //                     ? A[globalRow * K + globalCol]
    //                     : 0.f;
    //             }
    //         }
    //
    //         //
    //         // Load B tile to shared memory
    //         //
    //         for (int i = ty; i < BK; i += blockDim.y)
    //         {
    //             for (int j = tx; j < BN; j += blockDim.x)
    //             {
    //                 int globalRow = k0 + i;
    //                 int globalCol = blockCol + j;
    //
    //                 Bs[i][j] =
    //                     (globalRow < K && globalCol < N)
    //                     ? B[globalRow * N + globalCol]
    //                     : 0.f;
    //             }
    //         }
    //
    //         __syncthreads();
    //
    //         //
    //         // Compute on shared tiles
    //         //
    //         #pragma unroll
    //         for (int k = 0; k < BK; ++k)
    //         {
    //             float aFrag[TM];
    //             float bFrag[TN];
    //
    //             // load A fragment to registers
    //             #pragma unroll
    //             for (int i = 0; i < TM; ++i)
    //                 aFrag[i] = As[threadRow + i][k];
    //
    //             // load B fragment to registers
    //             #pragma unroll
    //             for (int j = 0; j < TN; ++j)
    //                 bFrag[j] = Bs[k][threadCol + j];
    //
    //             // outer product update
    //             #pragma unroll
    //             for (int i = 0; i < TM; ++i)
    //             {
    //                 #pragma unroll
    //                 for (int j = 0; j < TN; ++j)
    //                 {
    //                     acc[i][j] += aFrag[i] * bFrag[j];
    //                 }
    //             }
    //         }
    //
    //         __syncthreads();
    //     }
    //
    //     //
    //     // Store results
    //     //
    //     #pragma unroll
    //     for (int i = 0; i < TM; ++i)
    //     {
    //         int globalRow = blockRow + threadRow + i;
    //
    //         if (globalRow < M)
    //         {
    //             #pragma unroll
    //             for (int j = 0; j < TN; ++j)
    //             {
    //                 int globalCol = blockCol + threadCol + j;
    //
    //                 if (globalCol < N)
    //                     C[globalRow * N + globalCol] = acc[i][j];
    //             }
    //         }
    //     }
    // }
}
