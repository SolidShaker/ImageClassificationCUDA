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

        __shared__ float tileB[TILE_SIZE * 4][TILE_SIZE];

        const float4* A4 = reinterpret_cast<const float4*>(A);

        float sum = 0.0f;

        int K4 = (K + 3) / 4;
        int numTiles = (K4 + TILE_SIZE - 1) / TILE_SIZE;

        for (int k0 = 0; k0 < numTiles; ++k0)
        {
            //--------------------------------------------------
            // Load A tile (vectorized)
            //--------------------------------------------------
            int aCol4 = k0 * TILE_SIZE + tx;

            tileA[ty][tx] =
                (row < M && aCol4 < K4)
                ? A4[row * K4 + aCol4]
                : make_float4(0.f, 0.f, 0.f, 0.f);

            //--------------------------------------------------
            // Load B tile (scalar)
            //
            // Need TILE_SIZE * 4 rows because each float4 in A
            // corresponds to 4 scalar K positions
            //--------------------------------------------------
            for (int i = 0; i < 4; ++i)
            {
                int bRow = k0 * TILE_SIZE * 4 + ty * 4 + i;

                tileB[ty * 4 + i][tx] =
                    (bRow < K && col < N)
                    ? B[bRow * N + col]
                    : 0.0f;
            }

            __syncthreads();

            //--------------------------------------------------
            // Compute
            //--------------------------------------------------
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k)
            {
                float4 a = tileA[ty][k];

                int base = k * 4;

                sum += a.x * tileB[base + 0][tx];
                sum += a.y * tileB[base + 1][tx];
                sum += a.z * tileB[base + 2][tx];
                sum += a.w * tileB[base + 3][tx];
            }

            __syncthreads();
        }

        //------------------------------------------------------
        // Store result
        //------------------------------------------------------
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
