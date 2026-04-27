#include "DotProduct.cuh"




namespace VB
{
    __global__ void krDotStart(const float* __restrict__ A,     
                               const float* __restrict__ B,
                               float* __restrict__ blockSum,
                               int N)
    {
        const int tx = threadIdx.x;

        const int lane   = tx % TILE_SIZE;
        const int warpId = tx / TILE_SIZE;

        const int stride = blockIdx.x * blockDim.x + tx;

        const int idx = blockIdx.x * blockDim.x + tx;

        __shared__ float warpSum[TILE_SIZE];


        const float4 *A4 = reinterpret_cast<float4*>(A);
        const float4 *B4 = reinterpret_cast<float4*>(B);

        float sum = 0.f;

        while (idx < N / 4)
        {
            float4 a = A4[idx];
            float4 b = B4[idx];
            
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

            idx += stride;
        }

        sum = VB::krShflReduction(sum);
        
        if (lane == 0)
            warpSum[warpId] = sum;
        
        __syncthreads();

        if (warpId == 0)
        {
            sum = (lane < (blockDim.x + TILE_SIZE) / TILE_SIZE) 
                ? warpSum[lane] : 0.f;

            sum = VB::krShflReduction(sum);

            if (lane == 0)
                blockSum[blockIdx.x] = sum;
        }
    }

    __global__ void krDotEnd(const float* __restrict__ blockSum,
                             float* __restrict__ result,
                             int N)
    {
        const int tx = threadIdx.x; 

        const int lane   = tid % TILE_SIZE; 
        const int warpId = tid / TILE_SIZE;

        __shared__ float shared[TILE_SIZE]; 

        float sum = 0.0f; 

        for (int i = tx; i < N; i += blockDim.x) 
            sum += blockSum[i]; 

        sum = VB::krShflReduction(sum);

        if (lane == 0) 
            shared[warpId] = sum; 

        __syncthreads(); 

        if (warpId == 0)
        {
            sum = (lane < (blockDim.x + TILE_SIZE) / TILE_SIZE) 
                ? shared[lane] : 0.f; 

            sum = VB::krShflReduction(sum);

            if (lane == 0) 
                *result = sum; 
        }
    }
}

