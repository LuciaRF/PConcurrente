#include "utils.h"

#define THREADS_PER_BLOCK 8

__global__ //kernel
void rgba_to_greyscale(const uchar4* const rgbaImage,
    unsigned char* const greyImage,
    int numRows, int numCols)
{
    //TODO
    //Use 2D topology for threads and blocks
    // .x -> R ; .y -> G ; .z -> B ; .w -> A
    //output = .299f * R + .587f * G + .114f * B;

    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    int icol = blockIdx.x * blockDim.x + threadIdx.x;

    if (irow >= numRows || icol >= numCols) return;

    float res = 0.299f * rgbaImage[irow * numCols + icol].x + 0.587f * rgbaImage[irow * numCols + icol].y +
        0.114f * rgbaImage[irow * numCols + icol].z;

    greyImage[irow * numCols + icol] = res;


}

void your_rgba_to_greyscale(const uchar4* const h_rgbaImage, uchar4* const d_rgbaImage, //llamada main
    unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
    //You must fill in the correct sizes for the blockSize and gridSize
    //currently only one block with one thread is being launched

    float sizeCols = ceil(numCols / THREADS_PER_BLOCK)+1; 
    //float sizeRows = ceil(x * numRows / THREADS_PER_BLOCK)+ 1; para varias imagenes de diferentes tamaños se debería multiplicar por un número de forma que abarque todos los pixels
    float sizeRows = ceil(numRows / THREADS_PER_BLOCK)+ 1;

    const dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    const dim3 gridSize(sizeCols,sizeRows);

    //printf("numCols:%d - col %f,numRows %d - row %f",numCols, sizeCols,numRows,sizeRows);

    rgba_to_greyscale << <gridSize, blockSize >> > (d_rgbaImage, d_greyImage, numRows, numCols);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
