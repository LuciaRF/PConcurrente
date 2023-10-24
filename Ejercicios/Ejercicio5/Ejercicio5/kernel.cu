#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <time.h>

#include <stdio.h>

#define THREADS_PER_BLOCK 4

__global__ void thread_add(float* A, int n)
{

    int id = threadIdx.x; //identificador local por cada bloque de n tiene que sumar 1
    //Add 1 to each A[thread]

    atomicAdd(&A[id], 1);
    //A[id] = A[id] + 1;
}

int main()
{
    int n = 8;
    int numbytes = n * sizeof(float);

    float* h_A;
    float* d_A;

    //Create array A (10 in all index)

    h_A = (float*)malloc(numbytes);
    cudaMalloc(&d_A, numbytes);

    for (int i = 0; i < n; i++) {
        h_A[i] = 10;
    }

    cudaMemcpy(d_A, h_A, numbytes, cudaMemcpyHostToDevice);

    //Planificate grid 

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(n / THREADS_PER_BLOCK);

    //Run kernel using n threads

    thread_add << <gridDim, blockDim >> > (d_A, n);

    cudaMemcpy(h_A, d_A, numbytes, cudaMemcpyDeviceToHost);


    // Print results

    for (int i = 0; i < n; i++) {
        printf("%f\n", h_A[i]);
    }


    cudaFree(d_A);
    free(h_A);
}