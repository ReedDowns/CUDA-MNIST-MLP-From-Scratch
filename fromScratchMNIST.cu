#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h> // Still not sure why we include this; perhaps cublas compatability?

// call refers to the output of the function call...
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

// Choppy [0, 1] random number generator to fill + initialize matrices
void randInit(float* A, size_t n) { 
    for (size_t i = 0; i < n; i++) {
        A[i]=(float)rand()/(float)RAND_MAX;
    }
}

// Print matrix
void printMat (float *A, int D1, int D2) {
    for (int i = 0; i < D1; i++) {
        for (int j = 0; j < D2; j++) 
            printf("%8.3f", A[i * D2 + j]);
        printf("\n");
        }
    printf("\n");
}


// Input, Hidden nodes, output, depth,
#define I 784
#define H 128
#define O 10
#define D 2
#define BS 1 // Batch Size

void main() {
    // Hyperparameters?


    // Initialize arrays on host (before passing to device).

    // ***Set up foward prop and backward prop***
    cublasHandle_t handle; 
    CHECK_CUBLAS(cublasCreate(&handle));


    // Read in data

    // Apply forward prop and backward prop


    // Use streams for batch training? 


    cublasDestroy(&handle);
}