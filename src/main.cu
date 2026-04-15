#include <iostream>

#include "vibeflowTypes.cuh"
#include "vibeflow.cuh"

#include <iostream>

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
    int M = 1 << 10;
    int N = 1 << 5;
    int P = 1 << 10;
    int pP = VB::UTILS::HOST::GetPadding(P);

    size_t sizeA  = M * N;
    size_t sizeB  = N * P;
    size_t pSizeB = N * pP;
    size_t pSizeC = M * pP;

    float *hA = new float[sizeA];
    float *hB = new float[sizeB];
    float *hC = new float[pSizeC];

    for (size_t i = 0; i < sizeA; i++) hA[i] = 1.0f;
    for (size_t i = 0; i < sizeB; i++) hB[i] = 2.0f;

    VB::Variable<float> dA;
    VB::Variable<float> dB;
    VB::Variable<float> dC;

    dA.Allocate(sizeA);
    dB.Allocate(pSizeB);
    dC.Allocate(pSizeC);

    dA.Write(hA, sizeA);
    dB.Write(hB, sizeB);

    VB::DEVICE::GEMM::GEMM(dA, dB, dC, M, N, P, pP, VB::THREADS::T256); 
    CUDA_CHECK("An error occured in GEMM");
    cudaDeviceSynchronize();

    dC.Read(hC, pSizeC);

    for (int i = 0; i < 10; i++)
        std::cout << hC[i] << " ";
    std::cout << "\n expected =" << 2.0f * N;

    return 0;
}

