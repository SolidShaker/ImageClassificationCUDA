#include "vibeflow.cuh"


namespace VB
{
    __global__ void krDot(float *A, float *B, float *C, int N)
    {
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        int stride = blockDim.x * gridDim.x;

        float sum = 0.0f;

        // -------------------------
        // grid-stride loop
        // -------------------------
        while (idx < N)
        {
            sum += A[idx] * B[idx];
            idx += stride;
        }

        // -------------------------
        // warp reduction
        // -------------------------
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        int lane   = tid % 32;
        int warpId = tid / 32;

        __shared__ float warpSum[32];

        if (lane == 0)
            warpSum[warpId] = sum;

        __syncthreads();

        // -------------------------
        // final warp reduction
        // -------------------------
        if (warpId == 0)
        {
            sum = (lane < (blockDim.x + 31) / 32) ? warpSum[lane] : 0.0f;

            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, offset);

            if (lane == 0)
                atomicAdd(C, sum);
        }
    }
}
