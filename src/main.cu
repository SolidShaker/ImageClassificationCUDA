#include <VbTypes.cuh>
#include <VbTimer.cuh>
#include <VbFunctions/MatMul.cuh>
#include <VbFunctions/Utils.cuh>

#include <cuda_runtime.h>

#include <iostream>

void cpuGEMM(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void testPattern(float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = (float)i;
}

void runTestKernel(float* dC, int N)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    testPattern<<<blocks, threads>>>(dC, N);
    cudaDeviceSynchronize();
}

int main()
{
    int N = 1000;

    float* hC = new float[N];
    float* dC;

    cudaMalloc(&dC, N * sizeof(float));

    runTestKernel(dC, N);

    cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)
        std::cout << hC[i] << " ";

    std::cout << std::endl;

    cudaFree(dC);
    delete[] hC;
}


//
// int main()
// {
//     Vb::Timer gpuClock;
//     int M = 2500;
//     int N = 2505;
//     int K = 2502;
//
//     int pM= Vb::GetPadding(M, 4);
//     int pN= Vb::GetPadding(N, 4);
//     int pK= Vb::GetPadding(K, 4);
//
//     size_t sizeA = M * K;
//     size_t sizeB = K * N;
//     size_t sizeC = M * N;
//
//     float* hA = new float[sizeA];
//     float* hB = new float[sizeB];
//     float* hC = new float[sizeC];
//
//     for (size_t i = 0; i < sizeA; i++) hA[i] = 1.0f;
//     for (size_t i = 0; i < sizeB; i++) hB[i] = 2.0f;
//
//     Vb::Variable<float> dA;
//     Vb::Variable<float> dB;
//     Vb::Variable<float> dC;
//
//     dA.Allocate(pM * pK);
//     dB.Allocate(pK * pN);
//     dC.Allocate(pM * pN);
//
//     dA.Write(hA, sizeA);
//     dB.Write(hB, sizeB);
//
//     gpuClock.StartRecord();
//
//     Vb::GEMM(dA.Get(), dB.Get(), dC.Get(), M, N, K);
//     CUDA_CHECK("An error occured in GEMM");
//     cudaDeviceSynchronize();
//
//     gpuClock.StopRecord();
//
//     dC.Read(hC, sizeC);
//
//     std::cout << "GPU Time: " << gpuClock.GetTime() << "\n";
//     std::cout << "M_padded=" << M << " N_padded=" << N << " K_padded=" << K << "\n";
//     for (int i = 0; i < 10; i++)
//         std::cout << hC[i] << " ";
//     std::cout << "\n expected =" << 2.0f * K;
//
//     float* hC_cpu = new float[sizeC];
//     cpuGEMM(hA, hB, hC_cpu, M, N, K);
//
//     float maxError = 0.0f;
//     int badCount = 0;
//
//     for (int i = 0; i < M * N; i++)
//     {
//         float diff = fabs(hC[i] - hC_cpu[i]);
//
//         maxError = std::max(maxError, diff);
//
//         if (diff > 1e-3f)
//             badCount++;
//     }
//
//     std::cout << "Max error: " << maxError << std::endl;
//     std::cout << "Bad elements: " << badCount << " / " << (M * N) << std::endl;    
//
//     return 0;
// }

