#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <time.h>

#include <stdio.h>

#define THREADS_PER_BLOCK 32

__global__ void sync_sum(float* A, int n)
{
    //Sum A[thread] and A[thread + 1] into A[thread]. Considers a cyclic array (A[n] = A[0])

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        
        float own = A[id];
        float next = A[(id + 1) % n];

        __syncthreads();

        A[id] = own + next;
    }
}

int main()
{
    int n = 32;
    int numbytes = n * sizeof(float);

    srand(time(NULL));

    float* h_A;
    float* d_A;

    //Create array A (1, 2, 3,...,n)

    h_A = (float*)malloc(numbytes);

    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i;
    }

    cudaMalloc(&d_A, numbytes);
    cudaMemcpy(d_A, h_A, numbytes, cudaMemcpyHostToDevice);
    //Planificate grid 

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(n / THREADS_PER_BLOCK); // solo un bloque de threads

    //Run kernel using n threads

    sync_sum << <gridDim, blockDim >> > (d_A, n);

     // Print results
        cudaMemcpy(h_A, d_A, numbytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("CPU: %f - GPU: %f\n", (float)(i + ((i + 1) % n)), h_A[i]);
    }

}