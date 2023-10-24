#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>


__global__ void print(float* A, int n) {
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%d\n", A[i]); // No es lo habitual, cada uno de los threads accede a una posicion
    }
    printf("\n");
}
int main()
{
    int n = 16;
    int numbytes = n * sizeof(float);

    float* h_A = NULL;
    float* d_A = NULL;

    //Reserve memory

    h_A = (float*)malloc(numbytes);

    cudaMalloc((void**)&d_A, numbytes);

    //Set d_A to 0

    cudaMemset(d_A, 0, numbytes);

    //Copy data from d_A to h_A

    cudaMemcpy(h_A, d_A, numbytes, cudaMemcpyDeviceToHost);

   print <<<1,1 >>>(d_A, n);

    cudaGetLastError();

    //Print result

    for (int i = 0; i < n; i++) {
        printf("%f\n", h_A[i]);
    }
    //Free memory

    free(h_A);
    cudaFree(d_A);
  

    return 0;
}



