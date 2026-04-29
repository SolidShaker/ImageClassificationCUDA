#include <VbTypes.cuh>
#include <VbTimer.cuh>
#include <VbFunctions/MatMul.cuh>
#include <VbFunctions/Utils.cuh>

#include <cuda_runtime.h>

#include <iostream>




int main()
{
    Vb::Timer gpuClock;
    int M = 2500;
    int N = 2505;
    int K = 2502;

    int pM= Vb::GetPadding(M, 4);
    int pN= Vb::GetPadding(N, 4);
    int pK= Vb::GetPadding(K, 4);

    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    float* hA = new float[sizeA];
    float* hB = new float[sizeB];
    float* hC = new float[sizeC];

    for (size_t i = 0; i < sizeA; i++) hA[i] = 1.0f;
    for (size_t i = 0; i < sizeB; i++) hB[i] = 2.0f;

    Vb::Variable<float> dA;
    Vb::Variable<float> dB;
    Vb::Variable<float> dC;

    dA.Allocate(pM * pK);
    dB.Allocate(pK * pN);
    dC.Allocate(pM * pN);

    dA.Write(hA, sizeA);
    dB.Write(hB, sizeB);

    gpuClock.StartRecord();

    Vb::GEMM(dA.Get(), dB.Get(), dC.Get(), M, N, K);
    CUDA_CHECK("An error occured in GEMM");
    cudaDeviceSynchronize();

    gpuClock.StopRecord();

    dC.Read(hC, sizeC);

    std::cout << "GPU Time: " << gpuClock.GetTime() << "\n";
    std::cout << "M_padded=" << M << " N_padded=" << N << " K_padded=" << K << "\n";
    for (int i = 0; i < 10; i++)
        std::cout << hC[i] << " ";
    std::cout << "\n expected =" << 2.0f * K;

    return 0;
}

