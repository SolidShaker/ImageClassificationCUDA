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
    int M = 2500;
    int N = 1505;
    int P = 2502;

    int pM= VB::UTILS::HOST::GetPadding(M, 4);
    int pN= VB::UTILS::HOST::GetPadding(N, 4);
    int pP= VB::UTILS::HOST::GetPadding(P, 4);

    size_t sizeA = M * N;
    size_t sizeB = N * P;
    size_t sizeC = M * P;

    float *hA = new float[sizeA];
    float *hB = new float[sizeB];
    float *hC = new float[sizeC];

    for (size_t i = 0; i < sizeA; i++) hA[i] = 1.0f;
    for (size_t i = 0; i < sizeB; i++) hB[i] = 2.0f;

    VB::Variable<float> dA;
    VB::Variable<float> dB;
    VB::Variable<float> dC;

    dA.Allocate(pM * pN);
    dB.Allocate(pN * pP);
    dC.Allocate(pM * pP);

    dA.Write(hA, sizeA);
    dB.Write(hB, sizeB);

    VB::DEVICE::GEMM::GEMM(dA.Get(), pM, dB.Get(), pN, dC.Get(), pP, M, N, P);
    CUDA_CHECK("An error occured in GEMM");
    cudaDeviceSynchronize();

    dC.Read(hC, sizeC);

    std::cout << "M_padded=" << M << " N_padded=" << N << " P_padded=" << P << "\n";
    for (int i = 0; i < 10; i++)
        std::cout << hC[i] << " ";
    std::cout << "\n expected =" << 2.0f * N;

    return 0;
}

