#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define THREADS_PER_BLOCK 16

__global__ void staticReverse(int* d, int n)
{
    //Reverse vector using static shared memory
    int id = threadIdx.x;  //local  lo guardo cuando voy a acceder  a memoria  compartida
    int gid = blockDim.x * blockIdx.x + threadIdx.x;//identificador global
    int r_id = n - 1 - gid;

    __shared__ int sh[THREADS_PER_BLOCK];

    sh[id] = d[r_id]; //guardamos la info en la memoria compartida
  
    __syncthreads();//como  no sé en que orden va a  venir se pone sync para que lleguen todos hasta la memoria compartida se copie todos

    d[gid] = sh[id];
}

__global__ void dynamicReverse(int* d, int n)
{
    //Reverse vector using dynamic shared memory
    extern __shared__ int sh[]; //cambia la reserva de memoria se la doy en el kernel
 
    int id = threadIdx.x;  //local  lo guardo cuando voy a acceder  a memoria  compartida
    int gid = blockDim.x * blockIdx.x + threadIdx.x;//identificador global
    int r_id = n - 1 - gid;

    sh[id] = d[r_id]; //guardamos la info en la memoria compartida

    __syncthreads();//como  no sé en que orden va a  venir se pone sync para que lleguen todos hasta la memoria compartida se copie todos

    d[gid] = sh[id];
}

int main(void)
{
    const int n = 64;
    int a[n], r[n], d[n];

    for (int i = 0; i < n; i++) {
        a[i] = i; //input
        r[i] = n - i - 1; //reference
        d[i] = 0; //output
    }

    int* d_d;
    // Allocate memory d_d
    cudaMalloc(&d_d, n * sizeof(int));


    // run version with static shared memory

    //// Copy memory from a to d_d
    cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice); //a es el input que se tiene el dato de entradad

    staticReverse << <n/THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_d, n);
    //// Copy memory from d_d

    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

    // run dynamic shared memory version

    //// Copy memory from a to d_d
    cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);

    size_t sh_mem = sizeof(int) * THREADS_PER_BLOCK;
    dynamicReverse << <n/THREADS_PER_BLOCK, THREADS_PER_BLOCK, sh_mem >> > (d_d, n);
    //// Copy memory from d_d

    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        if (d[i] != r[i])
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

    cudaFree(d_d);
}
