#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <time.h>
#include <algorithm>

using namespace std;

//#define N 4096
#define N 256
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil_1d(int* in, int* out) {

    extern __shared__ int temp[];

    int g_index = blockDim.x * blockIdx.x + threadIdx.x;
    int l_index = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[l_index] = in[g_index];
    if (threadIdx.x < RADIUS) {
        temp[l_index - RADIUS] = in[g_index - RADIUS];
        temp[l_index + BLOCK_SIZE] = in[g_index + BLOCK_SIZE];
    }

    // Synchronize (ensure all the data is available)

    printf("g_index %d blockIdx.x %d threadIdx.x %d\n", g_index, blockIdx.x, threadIdx.x);
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        result += temp[l_index + offset];
    }
    // Store the result
    out[g_index] = result;
}

void fill_ints(int* x, int n) {
    fill_n(x, n, 1);
}

int main(void) {
    int* in, * out; // host copies of a, b, c
    int* d_in, * d_out; // device copies of a, b, c
    int size = (N + 2 * RADIUS) * sizeof(int);

    // Alloc space for host copies and setup values
    in = (int*)malloc(size);
    out = (int*)malloc(size);

    fill_ints(in, N + 2 * RADIUS);
    fill_ints(out, N + 2 * RADIUS);

    // Alloc space for device copies
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Copy to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

    // Launch stencil_1d() kernel on GPU

    int blockDim = BLOCK_SIZE;
    int gridDim = N / BLOCK_SIZE;

    int sh_size = (BLOCK_SIZE + 2 * RADIUS) * sizeof(int);

    stencil_1d << <gridDim, blockDim, sh_size >> > (d_in + RADIUS, d_out + RADIUS);

    // Copy result back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Error Checking
    for (int i = 0; i < N + 2 * RADIUS; i++) {
        if (i < RADIUS || i >= N + RADIUS) {
            if (out[i] != 1)
                printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
        }
        else {
            if (out[i] != 1 + 2 * RADIUS)
                printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2 * RADIUS);
        }
    }

    // Cleanup
    free(in); free(out);
    //free memory
    cudaFree(d_in);
    cudaFree(d_out);
    printf("Success!\n");
    return 0;
}