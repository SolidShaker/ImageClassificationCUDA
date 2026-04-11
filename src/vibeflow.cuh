#ifndef VIBEFLOW_H
#define VIBEFLOW_H

#include <cuda_runtime.h>




namespace VB
{
    namespace DOT
    {
        __global__ void krDotStart(float *A, float *B, float *blockSum, int N);
        __global__ void krDotEnd(float *blockSum, float *result, int N);
    }
}

#endif
