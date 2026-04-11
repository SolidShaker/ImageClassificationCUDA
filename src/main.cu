#include <iostream>
#include <cuda_runtime.h>


#define N (1<<10)

__global__ void 
krVecMul(float *A, float *B, float *C, float n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n)
        C[i] = A[i] * B[i];
}


int main()
{

    float *A, *B, *C;
    size_t size = N * sizeof(float);

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < N; i++)
    {
        A[i] = i * 1.0f;
        B[i] = 2.0f;
    }
    
    int device = 0;
    cudaGetDevice(&device);

    cudaMemAdvise(A, size, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(B, size, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(C, size, cudaMemAdviseSetPreferredLocation, device);

    cudaMemAdvise(A, size, cudaMemAdviseSetReadMostly, device);
    cudaMemAdvise(B, size, cudaMemAdviseSetReadMostly, device);

    cudaMemPrefetchAsync(A, size, device);
    cudaMemPrefetchAsync(B, size, device);
    cudaMemPrefetchAsync(C, size, device);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    krVecMul<<<blocks, threads>>>(A, B, C, N);

    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++)
        std::cout << A[i] << " * " << B[i] << " = " << C[i] << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
