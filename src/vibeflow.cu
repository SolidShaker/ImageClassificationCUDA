#include "vibeflow.cuh"


namespace VB::DOT
{
    __global__ void 
    krDotStart(float *A, float *B, float *blockSum, int N)
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

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        int lane   = tid % 32;
        int warpId = tid / 32;

        __shared__ float warpSum[32];

        if (lane == 0)
            warpSum[warpId] = sum;

        __syncthreads();

        if (warpId == 0)
        {
            sum = (lane < (blockDim.x + 31) / 32) ? warpSum[lane] : 0.0f;

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, offset);

            if (lane == 0)
                blockSum[blockIdx.x] = sum;
        }
    }

    __global__ void krDotEnd(float *blockSum, float *result, int N) 
    { 
        int tid = threadIdx.x; 
        float sum = 0.0f; 

        for (int i = tid; i < N; i += blockDim.x) 
            sum += blockSum[i]; 

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) 
            sum += __shfl_down_sync(0xffffffff, sum, offset); 


        __shared__ float shared[32]; 

        int lane = tid % 32; 
        int warp = tid / 32; 

        if (lane == 0) 
            shared[warp] = sum; 

        __syncthreads(); 

        if (warp == 0) 
        { 
            sum = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f; 

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) 
                sum += __shfl_down_sync(0xffffffff, sum, offset); 

            if (lane == 0) 
                *result = sum; 
        } 
    }
}
