#include <iostream>

#include "vibeflow.cuh"




int main()
{
    int N = 1000;
    float *A, *B, *C;
    size_t size = sizeof(float);
    size_t size_vec = N * size;

    cudaMallocManaged(&A, size_vec);
    cudaMallocManaged(&B, size_vec);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < N; i++)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    
    int device = 0;
    cudaSetDevice(device);

    cudaMemAdvise(A, size_vec, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(B, size_vec, cudaMemAdviseSetPreferredLocation, device);

    cudaMemAdvise(A, size_vec, cudaMemAdviseSetReadMostly, device);
    cudaMemAdvise(B, size_vec, cudaMemAdviseSetReadMostly, device);

    cudaMemPrefetchAsync(A, size_vec, device);
    cudaMemPrefetchAsync(B, size_vec, device);

    cudaMemPrefetchAsync(C, size, device);
    cudaMemset(C, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    VB::krDot<<<blocks, threads>>>(A, B, C, N);

    cudaDeviceSynchronize();

    std::cout << A[i] << " dot " << B[i] << " = " << *C << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
