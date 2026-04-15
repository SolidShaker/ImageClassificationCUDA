#ifndef VIBEFLOW_H
#define VIBEFLOW_H

#include <cuda_runtime.h>
#include "vibeflowTypes.cuh"

#include <stdexcept>




namespace VB
{
    namespace UTILS
    {
        namespace HOST
        {
            __host__ __forceinline__ int GetPadding(int size);
        }

        namespace DEVICE
        {
            __device__ __forceinline__ float krShflReduction(float var);
        }
    }

    namespace HOST
    {
    }

    namespace DEVICE
    {
        namespace DOT
        {
            __host__ void Dot(float *A, float *B, float *tem, float *result, 
                              int N, VB::THREADS threads);

            // REQUIRED: ZERO-PADDING
            __global__ void krDotStart(float *A, float *B, float *blockSum, 
                                       int N);
            __global__ void krDotEnd(float *blockSum, float *result, 
                                     int N);
        }

        // GEneral Matrix-to-matrix Multiplication
        namespace GEMM
        {
            __host__ void GEMM(float *A, float *B, float *C, 
                               int M, int N, int P, int pP, VB::THREADS threads);

            // REQUIRED: ZERO-PADDING
            __global__ void krGEMM(float *A, float *B, float *C, 
                                   int M, int N, int P, int pP);
        }
    }
}

#endif
