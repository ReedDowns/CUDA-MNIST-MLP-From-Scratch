// Test printing random numbers!
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 3
#define K 2
#define N 4

// Raw CUDA error handling
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) {\
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }\
}

// cuBLAS error handling
#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) {\
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// Print a matrix
void printMat (float *A, int D1, int D2) {
    for (int i = 0; i < D1; i++) {
        for (int j = 0; j < D2; j++) 
            printf("%8.3f", A[i * D2 + j]);
        printf("\n");
        }
    printf("\n");
}


int main() {
    
    // Initialize matrices
    float A[M * K] = {0.0f, 1.f, 2.f, 3.f, 4.f, 5.f};
    printf("Original matrix A:\n");
    printMat(A, M, K);

    float B[K*N] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    printf("Original matrix B:\n");
    printMat(B, K, N);

    float C[M*N] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    printf("Original matrix C:\n");
    printMat(C, M, N);

    /* 
    Compare matrix creation here to Example 1 in cuBLAS docs
    1) Is what they're doing safer than what I'm doing? Is what I'm doin
       safe enough?
    2) Should I be using malloc for my matrices? Is there a "right" way to
       declare pointers and typecast? ergo, parenthases and astrick '*' location 
    Related: Will things be freed upon program exit? What is 1-based and 2-based indexing?
    */
    
    // Create device handle 
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    /* Confirmed handle is called without pointing or dereferencing moving forward. */

    // Create device memory
    float *aD, *bD, *cD; // Haven't created host C yet; maybe don't need to...
    CHECK_CUDA(cudaMalloc(&aD, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bD, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&cD, M * N * sizeof(float)));

    // Copy A and B to device; 
    CHECK_CUDA(cudaMemcpy(aD,A,M * K *sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bD,B,K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cD,C,M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Check Sgemm inputs
    printf("M: %d, K: %d, N: %d \n\n", M, K, N);

    // Perform matrix multiplication using sGemm. No transpose
    float alpha=1.f, beta=1.f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             N, M, K, &alpha, // Cols B, Rows A, Shared
                             bD, N, aD, K,    // B^T, Cols B (rows B^T), A^T, rows A (cols A^T)
                             &beta, cD, N));

    
    CHECK_CUDA(cudaMemcpy(C, cD, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Non-Transpose AxB+C matMul:\n");
    printMat(C, M, N);

    // free(A); 
    // free(B); 
    // free(C);

    /* Why don't I need to free A, B, and C here? I know I didn't malloc() them, but
    they're still taking up memory. */

    CHECK_CUDA(cudaFree(aD));
    CHECK_CUDA(cudaFree(bD));
    CHECK_CUDA(cudaFree(cD));
    CHECK_CUBLAS(cublasDestroy(handle))
    
    return 0;
}

/*
1.
Row-major:
A = [1, 2, 3, 4
     5, 6, 7, 8]
Storage: [1, 2, 3, 4, 5, 6, 7, 8]

2.
Column-major: 
B = [1, 2, 3, 4,
     5, 6, 7, 8]
storage: [1, 5, 2, 6, 3, 7, 4, 8]

3.
Eliot's column-major example:
C = [1, 5,
     2, 6,
     3, 7, 
     4, 8]
Storage: [1, 5, 2, 6, 3, 7, 4, 8]
How I think C should be stored:  [1, 2, 3, 4, 5, 6, 7, 8] -> Correct but...

The important take away is that the 2D->1D for matrix C in C is the same as the 
2D->1D for matrix A in Fortran.
Transpose: 
AT = 2x3 (rows), 3x2 (cols)
AT = [0, 2, 4,
      1, 3, 5]
BT= 4x2 (rows), 2x4 cols
BT = [0, 4,
      1, 5,
      2, 6
      3, 7]
CT = BT x AT
CT = 4x3 (rows)
CT = [4, 12, 20,
      5, 17, 29,
      6, 22, 38,
      7, 26, 47]
C = [ 4,  5,  6,  7,
     12, 17, 22, 26
     20, 29, 38, 47]
*/


/* 
New mental approach: 
aD and bD are written as 
aD_Mem = [0, 1, 2, 3, 4, 5]
bD_Mem = [0, 1, 2, 3, 4, 5, 6, 7]

A, 3x2:
[0, 1
 2, 3
 4, 5]

However, ad_Memread as column-major 3x2:
[0, 2, 4,
 1, 3, 5]

B, column-major 2x4, 
[0, 4,
 1, 5,
 2, 6,
 3, 7]


*/