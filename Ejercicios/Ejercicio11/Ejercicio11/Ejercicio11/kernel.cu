
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

#define THREADS_PER_BLOCK 32
#define RANGE 18
#define MAX_AGE 72

__global__ void histogram_kernel(int* d_bins, int* d_in, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int myElement = d_in[tid];
    int binIndex = myElement / RANGE;
    atomicAdd(&(d_bins[binIndex]), 1);
}

void histogram_CPU(int* bins, int* data, int size, int bin_count) {
    for (int i = 0; i < bin_count; i++) {
        bins[i] = 0;
    }

    for (int i = 0; i < size; i++) {
        bins[data[i] / RANGE]++;
    }
}

int main()
{
    int n = 256;

    int* h_in = (int*)malloc(sizeof(int) * n);

    for (int i = 0; i < n; i++) {
        h_in[i] = rand() % MAX_AGE;
    }

    int bin_count = MAX_AGE / RANGE;
    int* h_ref_bins = (int*)malloc(sizeof(int) * bin_count);
    histogram_CPU(h_ref_bins, h_in, n, bin_count);

    int* d_in;
    //Allocate and copy memory for input
    cudaMalloc(&d_in, sizeof(int) * n);
    cudaMemcpy(d_in, h_in, sizeof(int) * n, cudaMemcpyHostToDevice);

    int* d_bins;
    //Allocate and set memory for output
    cudaMalloc(&d_bins, sizeof(int) * bin_count);
    cudaMemset(d_bins, 0, sizeof(int) * bin_count);


    //Configure kernel launch
    int blockDim = THREADS_PER_BLOCK;
    int gridDim = (n / blockDim) + 1;

    histogram_kernel << < gridDim, blockDim >> > (d_bins, d_in, n);

    int* h_bins = (int*)malloc(sizeof(int) * bin_count);

    //Copy output to CPU
    cudaMemcpy(h_bins, d_bins, sizeof(int) * bin_count, cudaMemcpyDeviceToHost);


    for (int i = 0; i < bin_count; i++) {
        if (h_ref_bins[i] != h_bins[i]) {
            printf("Error: %d - ref: %d\n", h_bins[i], h_ref_bins[i]);
        }
    }

    printf("Finished\n");
    printf("Ages:\n");
    for (int i = 0; i < bin_count; i++) {
        printf("%d - %d years old: ", i * RANGE, i * RANGE + RANGE);
        printf("%d\n", h_bins[i]);
    }

    return 0;
}
