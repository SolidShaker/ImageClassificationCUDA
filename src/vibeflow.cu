#include "vibeflow.cuh"

namespace VB::UTILS
{
    namespace HOST
    {
        __host__ __forceinline__ int
        GetPadding(int size)
        {
            return size % 4 == 0 ? size : (size + 3) & ~3;
        }
    }

    namespace DEVICE
    {
        __device__ __forceinline__ float
        krShflReduction(float var)
        {
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                var += __shfl_down_sync(0xffffffff, var, offset);
            return var;
        }
    }
}

namespace VB::DEVICE::DOT
{
    __host__ void 
    Dot(float *A, float *B, float *tem, float *result, 
        int N, VB::THREADS threads)
    {
        int thrs = VB::ToThreads(threads);

        if (thrs % 32 != 0)
            throw std::runtime_error("Threads must be multiple of 32");

        int N4 = N / 4;

        dim3 blockDim(thrs, 1);
        dim3 gridDim(
                (N4 + thrs - 1) / thrs
        );

        krDotStart<<<gridDim, blockDim>>>(A, B, tem, N);
        krDotEnd<<<1, thrs>>>(tem, result , gridDim.x);
    }

    __global__ void 
    krDotStart(float *A, float *B, float *blockSum, 
               int N)
    {

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        int stride = blockDim.x * gridDim.x;

        float4 *A4 = reinterpret_cast<float4*>(A);
        float4 *B4 = reinterpret_cast<float4*>(B);


        float sum = 0.0f;

        while (idx < N / 4)
        {
            float4 a = A4[idx];
            float4 b = B4[idx];

            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

            idx += stride;
        }

        sum = VB::UTILS::krShflReduction(sum);

        int lane   = tid % 32;
        int warpId = tid / 32;

        __shared__ float warpSum[32];

        if (lane == 0)
            warpSum[warpId] = sum;

        __syncthreads();

        if (warpId == 0)
        {
            sum = (lane < (blockDim.x + 31) / 32) ? warpSum[lane] : 0.0f;

            sum = VB::UTILS::krShflReduction(sum);

            if (lane == 0)
                blockSum[blockIdx.x] = sum;
        }
    }

    __global__ void krDotEnd(float *blockSum, float *result, 
                             int N) 
    { 
        int tid = threadIdx.x; 
        float sum = 0.0f; 

        for (int i = tid; i < N; i += blockDim.x) 
            sum += blockSum[i]; 

        sum = VB::UTILS::krShflReduction(sum);


        __shared__ float shared[32]; 

        int lane = tid % 32; 
        int warp = tid / 32; 

        if (lane == 0) 
            shared[warp] = sum; 

        __syncthreads(); 

        if (warp == 0) 
        { 
            sum = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f; 

            sum = VB::UTILS::krShflReduction(sum);

            if (lane == 0) 
                *result = sum; 
        } 
    }
}


namespace VB::DEVICE::GEMM
{
    __host__ void 
    GEMM(float *A, float *B, float *C, 
         int M, int N, int P, int pP, VB::THREADS threads)
    {
        int thrs = ToThreads(threads);

        if (thrs % 32 != 0)
            throw std::runtime_error("Threads must be multiple of 32");

        int P4 = pP / 4;

        dim3 blockDim(thrs, 1);
        dim3 gridDim(
                (pP + TILE_N - 1) / TILE_N,
                (M + TILE_M - 1) / TILE_M
        );

        krMatMul<<<gridDim, blockDim>>>(A, B, C, M, N, P, pP);
    }

    __global__ void krGEMM(float* A, float* B, float* C,
                           int M, int N, int P, int pP)
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp = threadIdx.x / WARP_SIZE;

        int warpsPerRow = blockDim.x / WARP_SIZE;

        int warpRow = warp / warpsPerRow;
        int warpCol = warp % warpsPerRow;

        int row = blockIdx.y * TILE_M + warpRow;
        int col4 = (blockIdx.x * TILE_N + warpCol * WARP_SIZE + lane);

        const float4* B4 = reinterpret_cast<const float4*>(B);
        int P4 = pP / 4;

        float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;

        __shared__ float tileB[TILE_M][TILE_N * 4];

        for (int k = 0; k < N; k += TILE_N)
        {
            float4 b = B4[(k + warpRow) * P4 + (warpCol * WARP_SIZE + lane)];

            tileB[warpRow][lane * 4 + 0] = b.x;
            tileB[warpRow][lane * 4 + 1] = b.y;
            tileB[warpRow][lane * 4 + 2] = b.z;
            tileB[warpRow][lane * 4 + 3] = b.w;

            __syncthreads();

            #pragma unroll
            for (int i = 0; i < TILE_N; ++i)
            {
                float a = 0.f;

                if ((k + i) < N && lane == i % WARP_SIZE)
                    a = A[row * N + (k + i)];

                a = __shfl_sync(0xffffffff, a, i % WARP_SIZE);

                v0 += a * tileB[i][lane * 4 + 0];
                v1 += a * tileB[i][lane * 4 + 1];
                v2 += a * tileB[i][lane * 4 + 2];
                v3 += a * tileB[i][lane * 4 + 3];
            }

            __syncthreads();
        }

        if (row < M)
        {
            if (col4 + 3 < P)
            {
                C[row * P + col4 + 0] = v0;
                C[row * P + col4 + 1] = v1;
                C[row * P + col4 + 2] = v2;
                C[row * P + col4 + 3] = v3;
            }
        }
    }
}

