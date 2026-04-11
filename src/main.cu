#include <iostream>

#include "vibeflowTypes.cuh"
#include "vibeflow.cuh"


#define DEBUG

#ifdef DEBUG
    #define CUDA_CHECK(msg) do {                                   \
        cudaError_t err = cudaGetLastError();                     \
        if (err != cudaSuccess) {                                 \
            std::cout << msg << " error: "                        \
                    << cudaGetErrorString(err) << std::endl;    \
        }                                                          \
    } while(0)
#else
    #define CUDA_CHECK(msg) ...
#endif



int main()
{
    int N = 1 << 20;
    int N4 = N / 4;
    int threads = 256;
    int blocks  = (N4 + threads -1 ) / threads;


    VB::Variable<float> A, B;
    A.Allocate(N);
    B.Allocate(N);

    // float *A, *B;
    // cudaMalloc((void**)&A, N * sizeof(float));
    // cudaMalloc((void**)&B, N * sizeof(float));

    float *hA = new float[N];
    float *hB = new float[N];
    for (int i = 0; i < N; i++)
    {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    A.Write(hA, N);
    B.Write(hB, N);
    // cudaMemcpy(A, hA, N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(B, hB, N * sizeof(float), cudaMemcpyHostToDevice);


    VB::Variable<float> blockSum;
    VB::Variable<float> result;
    blockSum.Allocate(blocks);
    result.Allocate();
    // float *blockSum;
    // float *result;
    // cudaMalloc((void**)&blockSum, blocks * sizeof(float));
    // cudaMalloc((void**)&result, sizeof(float));

    
    VB::DOT::krDotStart<<<blocks, threads>>>(A.Get(), B.Get(), blockSum.Get(), N);
    cudaDeviceSynchronize();
    CUDA_CHECK("Start dot error");

    VB::DOT::krDotEnd<<<1, threads>>>(blockSum.Get(), result.Get(), blocks);
    cudaDeviceSynchronize();
    CUDA_CHECK("End dot error");

    float d_result;
    result.Read(&d_result, 1);
    // cudaMemcpy(&d_result, result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << d_result << std::endl;

    // cudaFree(A);
    // cudaFree(B);
    // cudaFree(blockSum);
    // cudaFree(result);

    free(hA);
    free(hB);

    return 0;
}

