#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <time.h>

#include <stdio.h>

#define THREADS_PER_BLOCK 32

__global__ void array_sum(float* A, float* B, float* C, int n)
{
    //sum A[thread] and B[thread] in C[thread]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {

        extern __shared__ float sh_A[]; //para memoria compartida solo 1, el otro se guarda a continuación creando un array el doble de grande
        float* sh_B = &sh_A[blockDim.x]; //direccion de memoria de sh_B cuando acaba

        sh_A[threadIdx.x] = A[id]; //se hace por bloque hilo local
       // sh_A[blockDim.x + threadIdx.x] = B[id];//array completo para el B
        sh_B[threadIdx.x] = B[id];


        //C[id] = A[id] + B[id];
        C[id] = sh_A[threadIdx.x] + sh_B[threadIdx.x];
    }
}

int main()
{
    int n = 32; //(test with 64, 128, ...)
    int numbytes = n * sizeof(float);

    srand(time(NULL));

    float* h_A;
    float* h_B;
    float* h_C;

    float* d_A;
    float* d_B;
    float* d_C;

    //Create array A (random)

    h_A = (float *)malloc(numbytes);
    cudaMalloc((void**)&d_A, numbytes);

    //Create array B (random)
    h_B = (float*)malloc(numbytes);
    cudaMalloc((void**)&d_B, numbytes);

    for (int i = 0; i < n; i++) {
        h_A[i] = rand() % 10 + 1;
        h_B[i] = rand() % 10 + 1;
    }

    cudaMemcpy(d_A, h_A, numbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numbytes, cudaMemcpyHostToDevice);

    //Initialize array C

    h_C = (float*)malloc(numbytes);
    cudaMalloc((void**)&d_C, numbytes);


    //Planificate grid (total threads per block = 32)

    dim3 dimBlock(THREADS_PER_BLOCK);
    dim3 dimGrid(ceil(n / (float)THREADS_PER_BLOCK));

    size_t sh_mem = sizeof(float) * THREADS_PER_BLOCK * 2;//para memoria compartida guardar el doble de memoria

    //Run kernel using n threads

    array_sum << < dimGrid, dimBlock, sh_mem >> > (d_A, d_B, d_C, n);

    // Print results

    cudaMemcpy(h_C, d_C, numbytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
}