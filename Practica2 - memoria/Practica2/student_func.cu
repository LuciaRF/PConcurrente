#include <stdio.h>
#include "utils.h"

using namespace std;


#define THREADS_PER_BLOCK 32
#define RADIUS 3


__device__ int clamp(int value, int minValue, int maxValue) {

    return min(max(value, minValue), maxValue);
}

__global__
void gaussian_blur(const unsigned char* const inputChannel,
    unsigned char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    if (gidx >= numCols || gidy >= numRows)
        return;

    int pos = gidy * numCols + gidx;
    int radioFilter = filterWidth / 2;
    

    // Declarar memoria compartida para el filtro y los valores de la imagen
    extern __shared__ float sharedFilter[];
    float* sharedImage = &sharedFilter[THREADS_PER_BLOCK + filterWidth];

    // Cargar los valores del filtro en la memoria compartida
    if (threadIdx.x < filterWidth && threadIdx.y < filterWidth) {
        int index = threadIdx.y * filterWidth + threadIdx.x;
        sharedFilter[index] = filter[index];
        printf("%d", sharedFilter[index]);
    }

        // Cargar los valores de la imagen en la memoria compartida
     int sharedPos = threadIdx.y * blockDim.x + threadIdx.x;
     sharedImage[sharedPos] = inputChannel[pos];

    __syncthreads();

    float blur = 0.0f;

    for (int i = -radioFilter; i <= radioFilter; i++) {
        for (int j = -radioFilter; j <= radioFilter; j++) {
            int x = clamp(gidx + j, 0, numCols - 1);
            int y = clamp(gidy + i, 0, numRows - 1);

            int xFilter = j + radioFilter;
            int yFilter = i + radioFilter;

            float filterValue = sharedFilter[yFilter * filterWidth + xFilter]; // Valor del filtro en memoria compartida
            float imageValue = sharedImage[(threadIdx.y + i) * numCols + (threadIdx.x + j)]; // Valor de la imagen en memoria compartida

            blur += filterValue * imageValue;
        }
    }

    outputChannel[pos] = blur;
}


__global__
void separateChannels(const uchar4* const inputImageRGBA,
    int numRows,
    int numCols,
    unsigned char* const redChannel,
    unsigned char* const greenChannel,
    unsigned char* const blueChannel)
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gidx >= numCols || gidy >= numRows) return;

    int pos = gidy * numCols + gidx;

    uchar4 pixel = inputImageRGBA[pos]; //para memoria global es mas eficiente 

    redChannel[pos] = pixel.x;
    greenChannel[pos] = pixel.y;
    blueChannel[pos] = pixel.z;
}


__global__
void recombineChannels(const unsigned char* const redChannel,
    const unsigned char* const greenChannel,
    const unsigned char* const blueChannel,
    uchar4* const outputImageRGBA,
    int numRows,
    int numCols)
{

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;

    int pos = gidy * numCols + gidx;

    uchar4 pixel;

    pixel.x = redChannel[pos];
    pixel.y = greenChannel[pos];
    pixel.z = blueChannel[pos];
    pixel.w = 0xFF;

    outputImageRGBA[pos] = pixel;

}

unsigned char* d_red, * d_green, * d_blue;
float* d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
    const float* const h_filter, const size_t filterWidth)
{
    cudaMalloc((void**)&d_red, sizeof(unsigned char) * numRowsImage * numColsImage);
    cudaMalloc((void**)&d_green, sizeof(unsigned char) * numRowsImage * numColsImage);
    cudaMalloc((void**)&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage);
    cudaMalloc((void**)&d_filter, sizeof(float) * filterWidth * filterWidth);
    cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice);
}

void your_gaussian_blur(const uchar4* const h_inputImageRGBA, uchar4* const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
    unsigned char* d_redBlurred,
    unsigned char* d_greenBlurred,
    unsigned char* d_blueBlurred,
    const int filterWidth)

{

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    int threadsperBlock = sqrt(props.maxThreadsPerBlock);

    //TODO complete

    const dim3 blockSize(threadsperBlock, threadsperBlock);
    const dim3 gridSize(ceil(numCols / (float)blockSize.x), ceil(numRows / (float)blockSize.y));

    //TODO Use Shared Memory when necessary

    separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA,
        numRows,
        numCols,
        d_red,
        d_green,
        d_blue);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    int sharedFilter = (filterWidth + THREADS_PER_BLOCK * 2) * sizeof(float);


    gaussian_blur << <gridSize, blockSize, sharedFilter >> > (
        d_red,
        d_redBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    gaussian_blur << <gridSize, blockSize, sharedFilter >> > (
        d_blue,
        d_blueBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    gaussian_blur << <gridSize, blockSize, sharedFilter >> > (
        d_green,
        d_greenBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    recombineChannels << <gridSize, blockSize >> > (d_redBlurred,
        d_greenBlurred,
        d_blueBlurred,
        d_outputImageRGBA,
        numRows,
        numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


void cleanup() {
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
    cudaFree(d_filter);

}

