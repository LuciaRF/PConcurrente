#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>

#define BLOCK_SIZE 1024

#define ARRAY_SIZE 1048576

__global__ void global_reduce_kernel(float* d_out, float* d_in)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	// loop for doing reduction in global mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {

		if (tid < s) {
			d_in[myId] += d_in[myId + s];
		}
		__syncthreads();
	}


	// only thread 0 writes result for this block back to global mem
	if (tid == 0) {
		d_out[blockIdx.x] = d_in[myId];
	}
}


__global__ void shmem_reduce_kernel(float* d_out, const float* d_in)
{
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	// load shared mem from global mem
	sdata[tid] = d_in[myId];

	// make sure entire block is loaded!

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {

		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}


	// only thread 0 writes result for this block back to global mem
	if (tid == 0) {
		d_out[blockIdx.x] = sdata[tid];
	}

}

void reduce(float* d_out, float* d_intermediate, float* d_in,
	int size, bool usesSharedMemory)
{
	// assumes that size is not greater than BLOCK_SIZE^2
	// and that size is a multiple of BLOCK_SIZE
	int threads = BLOCK_SIZE; //Complete
	int blocks = size / threads; //Complete
	if (usesSharedMemory)
	{
		shmem_reduce_kernel << <blocks, threads, threads * sizeof(float) >> >
			(d_intermediate, d_in);
	}
	else
	{
		global_reduce_kernel << <blocks, threads >> >
			(d_intermediate, d_in);
	}
	// now we're down to one block left, so reduce it
	threads = blocks; //Complete
	blocks = 1; //Complete
	if (usesSharedMemory)
	{
		shmem_reduce_kernel << <blocks, threads, threads * sizeof(float) >> >
			(d_out, d_intermediate);
	}
	else
	{
		global_reduce_kernel << <blocks, threads >> >
			(d_out, d_intermediate);
	}
}


int main() {

	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);


	float* h_in = (float*)malloc(ARRAY_BYTES);

	float ref = 0;

	srand(time(NULL));

	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = rand() % 10 + 1;
		ref += h_in[i];
	}

	float* d_in, * d_intermediate, * d_out;

	// allocate GPU memory
	cudaMalloc(&d_in, ARRAY_BYTES);
	cudaMalloc(&d_intermediate, sizeof(float) * BLOCK_SIZE);
	cudaMalloc(&d_out, sizeof(float));

	// copy array to GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);

	float h_out = 0;
	// get result from GPU
	cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

	printf("reference: %f, output: %f\n", ref, h_out);

	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_intermediate);

}