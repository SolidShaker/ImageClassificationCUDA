#ifndef VIBEFLOW_H
#define VIBEFLOW_H

#include <cuda_runtime.h>




namespace VB
{
    __global__ void krDot(float *A, float *B, float *C, int N);
}

#endif
