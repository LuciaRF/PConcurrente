#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>

#include <stdio.h>

#define THREADS_PER_BLOCK 4 

__global__ void hello_world(float* A, int n)
{
    // Print hello world + A[thread] for each thread
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("blockID: %d, threadId: %d\n", blockIdx.x, threadIdx.x);

    if (id < n) {
        printf("hello world %f\n", A[id]);
    }
}

int main()
{
    int n = 32;
    int numbytes = n * sizeof(float);

    float* h_A;
    float* d_A;

    //Reserve memory

    h_A = (float*)malloc(numbytes);
    cudaMalloc((void**)&d_A, numbytes);

    //Set initialize h_A to 1, 2, 3,...,n

    for (int i = 0; i < n; i++) {
        h_A[i] = i;
    }

    //Copy data from h_A to d_A

    cudaMemcpy(d_A, h_A, numbytes, cudaMemcpyHostToDevice);

    //Planificate gridDim and blockDim

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(ceil(n / (float) THREADS_PER_BLOCK));


    //Run kernel

    hello_world << <gridDim, blockDim >> > (d_A, n);

    //Print result

    //Free memory
    free(h_A);
    cudaFree(d_A);
}