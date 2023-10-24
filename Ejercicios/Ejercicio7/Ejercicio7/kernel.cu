#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <time.h>

#include <stdio.h>

#define THREADS_PER_BLOCK 16

__global__ void sync_sum(float* A, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= n) return;

    //Modify to use Shared Memory

    extern __shared__ float sh[];
    sh[threadIdx.x] = A[id]; //float own = A[id];
    if (threadIdx.x == blockIdx.x - 1) sh[(threadIdx.x + 1) % n] = A[(id + 1) % n]; //si  estoy en la ultima posicion de mi bloque en la ultima posicion guarda A[id +1]

    __syncthreads();

    A[id] = sh[threadIdx.x] + sh[threadIdx.x+1]; //float next = A[(id + 1) % n];
}

int main()
{
    int n = 32;
    int numbytes = n * sizeof(float);

    srand(time(NULL));

    float* h_A;

    float* d_A;

    h_A = (float*)malloc(numbytes);;

    cudaMalloc((void**)&d_A, numbytes);

    for (int i = 0; i < n; i++) {
        h_A[i] = i;
    }

    cudaMemcpy(d_A, h_A, numbytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(THREADS_PER_BLOCK);
    dim3 dimGrid(ceil(n / (float)dimBlock.x));


    //Add configuration to use Shared Memory

    size_t sh_mem = sizeof(float) * (THREADS_PER_BLOCK + 1); //+1 para guardar la variable extra

    sync_sum << <dimGrid, dimBlock, sh_mem >> > (d_A, n);

    cudaMemcpy(h_A, d_A, numbytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%f + %f = %f \n", (float)i, (float)((i + 1) % n), h_A[i]);
    }

    free(h_A);
    cudaFree(d_A);
}