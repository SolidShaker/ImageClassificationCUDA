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
    cudaGetDevice(&device);

    cudaMemAdvise(A, size_vec, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(B, size_vec, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(C, size, cudaMemAdviseSetPreferredLocation, device);

    cudaMemAdvise(A, size_vec, cudaMemAdviseSetReadMostly, device);
    cudaMemAdvise(B, size_vec, cudaMemAdviseSetReadMostly, device);

    cudaMemPrefetchAsync(A, size_vec, device);
    cudaMemPrefetchAsync(B, size_vec, device);
    cudaMemPrefetchAsync(C, size, device);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    VB::krDot<<<blocks, threads>>>(A, B, C, N);

    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++)
        std::cout << A[i] << " * " << B[i] << " = " << *C << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
