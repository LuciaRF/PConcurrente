#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <time.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 4

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);


void MatMul(const Matrix A, const Matrix B, Matrix C);

int main()
{
    int size = 4;

    Matrix A;
    Matrix B;
    Matrix C;

    A.width = size;
    B.width = size;
    C.width = size;

    A.height = size;
    B.height = size;
    C.height = size;

    A.elements = (float*)malloc(sizeof(float) * size * size);
    B.elements = (float*)malloc(sizeof(float) * size * size);
    C.elements = (float*)malloc(sizeof(float) * size * size);

    srand(time(NULL));

    for (int i = 0; i < size * size; i++) {
        A.elements[i] = 1;
        B.elements[i] = 1;
        C.elements[i] = 0;
    }

    MatMul(A, B, C);

    for (int i = 0; i < C.height; i++) {
        for (int j = 0; j < C.width; j++) {
            printf("%f ", C.elements[i * C.width + j]);
        }
        printf("\n");
    }

    return 0;
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory

    Matrix d_A, d_B, d_C;
    d_A.width = A.width;
    d_A.height = A.height;

    d_B.width = B.width;
    d_B.height = B.height;
    
    d_C.width = C.width;
    d_C.height = C.height;

    size_t numbytes = sizeof(float) * A.width * A.height;

    cudaMalloc(&d_A.elements, numbytes);
    cudaMalloc(&d_B.elements, numbytes);
    cudaMalloc(&d_C.elements, numbytes);

    cudaMemcpy(d_A.elements, A.elements, numbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, B.elements, numbytes, cudaMemcpyHostToDevice);

    // Allocate C in device memory

    // Invoke kernel

    dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridDim(C.width/blockDim.x, C.height / blockDim.y); //normalmente hacer el shape

    MatMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_C);

    // Read C from device memory

    cudaMemcpy(C.elements, d_C.elements, numbytes, cudaMemcpyDeviceToHost);

    // Free device memory

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into a temp variable

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= C.height || col >= C.width) return;

    int pos = row * C.width + col;

    C.elements[pos] = 0;
    float res = 0;// en lugar de C.elements para que no este grabando en memoria global
    for (int i = 0; i < A.width;i++) {
        res += A.elements[row * A.width + i] * B.elements[i * B.width + col];
    }
    C.elements[pos] = res;
}



