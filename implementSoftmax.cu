#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h> // Still not sure why we include this; perhaps cublas compatability?


// Call refers to the output of the function call...
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) {\
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }\
}
/*
Questions: 
1) call is an output from a CUDA op, which we redefine as err. Why redefine as err? 
   Why not just pass call to the test and to cudaGetErrorString()?
2) How do know what types the cuda error handling (and cublas status) are? Read API docs? 
   Or, probably read header files too.
3) Why can't I add comments at the end of the macro lines? 
*/

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) {\
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#define M 3
#define K 6
#define N 4

// print matrix
void printMat (float *a, int d1, int d2) {
    for (int i = 0; i < d1; i++) {
        for (int j = 0; j < d2; j++) 
            printf("%8.3f", a[i * d2 + j]);
        printf("\n");
        }
    printf("\n");
}


__global__ void softmax(
    // float* m1,  // Matrix 1
    float* m2,  // Matrix 2
    int n,      // Dimension one (weighted results, ergo rows)
    int b       // Dimension two (batch size, ergo columns)
    ) {
    // Assign y to row, x to column. Memory stored along y. Warps along x.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int id = row * b + col; // row number * number of elements in row + number of elements toward current column

    // printf("threadIdx.x/y: %d/%d\t blockDim.x/y: %d/%d\t row/col: %d/%d\n", 
            // threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col);
    /* blockIdx.x/y will be 1. So, row/column will be determined*/

    /* I should just transpose it all.. */
    
    /* Each thread will own a result. That thread needs to find its column, sum its column, 
       and replace its value with the exp(val)/sum */
    if (row <  n && col < b) {
        // m2[id] = m2[id] + m1[id];
         
        float sumExp=0.f;
        float sumCheck=0.f;

        float colMax = 0.f;
        for (int i = 0; i < n; i++) if (m2[i * b + col] > colMax) colMax = m2[i * b + col];

        for (int i = 0; i < n; i++) {
            sumCheck += m2[i * b + col];
            sumExp += expf(m2[i * b + col] - colMax);
            if (row == 1 && col == 4) printf("i: %d, m2[%.1f] sumExp: %.10f \n", i, m2[i], sumExp );
        }
        // printf("row:col:id %i:%i:%i, m2[%i] = %f, sumCheck=%f \n", row, col, id, id, m2[id], sumCheck);

        // if (row == 1 && col ==4)
        printf("m2[%d]: %.1f, expf(%.1f - %.1f): %f \n", id, m2[id], m2[id], colMax, expf(m2[id] - colMax));
        m2[id] = expf(m2[id] - colMax) / sumExp;
        // printf("sumExp=%f\n", sumExp);
        printf("New m2[%d]: %.12f \n", id, m2[id]);
    }

}

int main () {
    // Initialize matrices
    float A[M * K] = {0.0f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f};
    printf("Original matrix A:\n");
    printMat(A, M, K);

    float *dA;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * M * K));
    CHECK_CUDA(cudaMemcpy(dA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));

    dim3 blockDim(K, M); // M=3, K=6; trying to work with 3x6 matrix (GPU row,col -> y,x)
    dim3 gridDim(1);
    // Looking for one block, so grid should be fine.

    softmax<<<gridDim, blockDim>>>(dA, M, K);
    cudaDeviceSynchronize();
    
    CHECK_CUDA(cudaFree(dA));
    return 0;
}