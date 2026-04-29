#include <VbTypes.cuh>
#include <VbTimer.cuh>
#include <VbFunctions/MatMul.cuh>
#include <VbFunctions/Utils.cuh>

#include <test/testGEMM.cuh>

#include <cuda_runtime.h>

#include <iostream>




int main()
{
    Vb::Timer gpuClock;
    Vb::Timer gpuClocktest;
    int M = 2500;
    int N = 2505;
    int K = 2502;

    int pM= Vb::GetPadding(M, 2);
    int pN= Vb::GetPadding(N, 2);
    int pK= Vb::GetPadding(K, 2);

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

    Vb::Variable<float> dCtest;    
    dCtest.Allocate(pM * pN);

    float* hCtest = new float[sizeC];

    gpuClocktest.StartRecord();
    Test::GEMM(dA.Get(), dB.Get(), dCtest.Get(), M, N, K);
    CUDA_CHECK("An error occured in GEMM");
    cudaDeviceSynchronize();
    gpuClocktest.StopRecord();

    dCtest.Read(hCtest, sizeC);

    std::cout << "GPU Test Time: " << gpuClocktest.GetTime() << "\n";

    float maxError = 0.0f;
    int bad = 0;

    for (int i = 0; i < M * N; i++)
    {
        float diff = fabs(hC[i] - hCtest[i]);

        maxError = std::max(maxError, diff);

        if (diff > 1e-3f)
            bad++;
    }

    std::cout << "\n=== COMPARISON ===\n";
    std::cout << "Max error: " << maxError << "\n";
    std::cout << "Bad elements: " << bad << " / " << (M * N) << "\n";    

    return 0;
}

