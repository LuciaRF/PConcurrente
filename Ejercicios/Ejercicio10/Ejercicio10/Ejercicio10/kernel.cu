#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

#define THREADS_PER_BLOCK 32

__global__ void scan_kernel(int* d_data, int size) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= size)
        return;
    //TODO Complete scan algorithm
    for (int s = 1; s <= size; s <<= 1) {
		int spot = tid - s;

		int value = 0;
		if (spot >= 0)
			value = d_data[spot];
		__syncthreads();
		if (spot >= 0)
			d_data[tid] += value;
		__syncthreads();
	}
}

void scan_CPU(int* data, int size) {
    int acc = 0;
    for (int i = 0; i < size; i++) {
        acc = acc + data[i];
        data[i] = acc;
    }
}

int main()
{
    int n = 100;

    int* h_in = (int*)malloc(sizeof(int) * n);
    int* h_ref = (int*)malloc(sizeof(int) * n);

    for (int i = 0; i < n; i++) {
        h_in[i] = i;
        h_ref[i] = i;
    }

    scan_CPU(h_ref, n);

    int* d_data;
    //Allocate and copy memory
    cudaMalloc(&d_data, sizeof(int) * n);
    cudaMemcpy(d_data, h_in, sizeof(int) * n, cudaMemcpyHostToDevice);


    //Launch thread configure
    //int blockDim = THREADS_PER_BLOCK;
    //int gridDim = ceil(n /  (float) THREADS_PER_BLOCK);

    scan_kernel << < ceil(n / (float)THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (d_data, n);

    //Copy result to CPU
    cudaMemcpy(h_in, d_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        if (h_in[i] != h_ref[i]) {
            printf("Error: %d - ref: %d\n", h_in[i], h_ref[i]);
        }
    }

    printf("Finished\n");

    return 0;
}

