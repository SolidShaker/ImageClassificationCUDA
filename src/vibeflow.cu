#include "vibeflow.cuh"


namespace VB
{
    __global__ void 
    krDot(float *A, float *B, float *C, int N)
    {
        __shared__ float cache[256];

        int i = threadIdx.x;
        int pos = blockDim.x * blockIdx.x + threadIdx.x;

        float sum = 0;
        int stride_global = gridDim.x * blockDim.x;
        while (pos < N)
        {
            sum += A[pos] * B[pos];
            pos += stride_global;
        }
        cache[i] = sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (i < stride)
                cache[i] += cache[i + stride];
            __syncthreads();
        }
        
        if (threadIdx.x == 0)
            atomicAdd(C, cache[0]);
    }
}
