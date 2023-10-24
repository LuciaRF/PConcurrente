#include <stdio.h>
#include "utils.h"

#define THREADS_PER_BLOCK 8

__global__
void gaussian_blur(const unsigned char* const inputChannel,
    unsigned char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= numRows || col >= numCols) return;

    int index = row * numCols + col;

    int radio = filterWidth / 2;
    float res = 0.f;

    // Se realiza la convolucion teniendo en cuenta la fila y la columna del filtro que pase por todo el array de la imagen de entrada
    for (int offset_row = -radio; offset_row <= radio; offset_row++)
    {
        for (int offset_col = -radio; offset_col <= radio; offset_col++)
        {
            // Calculo del indice del pixel de la imagen + valor de offset para calcular la convolucion
             int image_row = row + offset_row;
             int image_col = col + offset_col;

             //excepciones para que no haya errores con los bordes
             if (image_row < 0)
                 image_row = 0;
             if (image_row >= numRows)
                 image_row = numRows - 1;
             if (image_col < 0)
                 image_col = 0;
             if (image_col >= numCols)
                 image_col = numCols - 1;

            // Calculo del indice del filtro
             int filter_pos = (offset_row + radio) * filterWidth + (offset_col + radio);
             float filter_val = filter[filter_pos];

            // Se guarda el resultado de la convolucion en un punto para la devolucion del valor completo a la imagen de salida
            int pos_suma = image_row * numCols + image_col;
            res += (inputChannel[pos_suma]) * filter_val;
        }
    }

    outputChannel[index] = res;
}


__global__
void separateChannels(const uchar4* const inputImageRGBA,
    int numRows,
    int numCols,
    unsigned char* const redChannel,
    unsigned char* const greenChannel,
    unsigned char* const blueChannel)
{
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    int icol = blockIdx.x * blockDim.x + threadIdx.x;

    if (irow >= numRows || icol >= numCols) return;

    redChannel[irow * numCols + icol] = inputImageRGBA[irow * numCols + icol].x;
    greenChannel[irow * numCols + icol] = inputImageRGBA[irow * numCols + icol].y;
    blueChannel[irow * numCols + icol] = inputImageRGBA[irow * numCols + icol].z;
}


__global__
void recombineChannels(const unsigned char* const redChannel,
    const unsigned char* const greenChannel,
    const unsigned char* const blueChannel,
    uchar4* const outputImageRGBA,
    int numRows,
    int numCols)
{
    
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    int icol = blockIdx.x * blockDim.x + threadIdx.x;

    if (irow >= numRows || icol >= numCols) return;

    outputImageRGBA[irow * numCols + icol].x = redChannel[irow * numCols + icol];
    outputImageRGBA[irow * numCols + icol].y = greenChannel[irow * numCols + icol];
    outputImageRGBA[irow * numCols + icol].z = blueChannel[irow * numCols + icol];


}

unsigned char* d_red, * d_green, * d_blue;
float* d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
    const float* const h_filter, const size_t filterWidth)
{
    cudaMalloc((void**)& d_red, sizeof(unsigned char)* numRowsImage* numColsImage);
    cudaMalloc((void**)& d_green, sizeof(unsigned char)* numRowsImage* numColsImage);
    cudaMalloc((void**)& d_blue, sizeof(unsigned char)* numRowsImage* numColsImage);
    cudaMalloc((void**)& d_filter, sizeof(float)* filterWidth * filterWidth);
    cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth,cudaMemcpyHostToDevice);
}

void your_gaussian_blur(const uchar4* const h_inputImageRGBA, uchar4* const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
    unsigned char* d_redBlurred,
    unsigned char* d_greenBlurred,
    unsigned char* d_blueBlurred,
    const int filterWidth)
{

    //TODO complete

    float sizeCols = ceil(numCols / THREADS_PER_BLOCK) + 1;
    float sizeRows = ceil(numRows / THREADS_PER_BLOCK) + 1;

    const dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    const dim3 gridSize(sizeCols, sizeRows);

    //TODO Use Shared Memory when necessary

    separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA,
        numRows,
        numCols,
        d_red,
        d_green,
        d_blue);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


    gaussian_blur << <gridSize, blockSize >> > (
        d_red,
        d_redBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    gaussian_blur << <gridSize, blockSize >> > (
        d_blue,
        d_blueBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    gaussian_blur << <gridSize, blockSize >> > (
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

