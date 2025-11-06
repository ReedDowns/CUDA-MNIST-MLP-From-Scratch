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

/*128x784*/ 
/*784x1  */ 
/*128x1  */ 
    // Set device memory, 
    // Run matmul 1: Input -> hl1
    // Activation
    // Run matmul 2: hl1 -> hl1
    // Activation
    // Run matmul 3:  hl2 -> output

void relu() {}
void softmat() {}

// Overwrite m2 with results from m1 + m2
__gobal__ void gpu_simple_add_2D (float* m1, float* m2, int *dim1, int *dim2) {

}

// Apply weights to input, 
// Pre-apply bias
float layer(
    cublasHandle_t handle,
    float* weights1, /*MxN*/ 
    float* input,  /*Nx1*/ 
    float* bias,    /*Mx1*/ 
    float* output,  /*Mx1*/
    int M, int K, int N,
    char activation,
)  {
    // Preload bias into output
    
    
    // Sgemm
    float alpha=1.f, beta=1.f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, input,N, weights1,K,&beta, output,N));
    
    /*This layer now stored in output (external)*/
    
    // Activation 
    if (activation == "relu") {relu(output);}
    if (activation == "softmax") {softmax(output);} 
    else {
        fprintf("No activation function in %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    /* 
    Output (or whatever was fed to output) is now activated, ready to pass to next layer.
    Shape should be Mx1 (or batch size; will need to determine later).
    */
}

void main() {
    // Hyperparameters?


    // Initialize arrays on host (before passing to device).
    /* Layer 0 - Input*/
    float* iplHost[I*BS];          // Input layer, 784x1 
    /* Layer 1 - Hidden 1 */
    float* weightsOneHost[H*I];     // Weights for layer one 128x784
    float* biasOneHost[H*BS];          // Bias for layer 128x1
    float* OPL_OneHost[H*BS];
    // OPL1: 128xBS
    /* Layer 2 - Hidden 2 */
    float* weightsTwoHost[H*H];     // Weights for layer two 128x128
    float* biasTwoHost[H*BS];          // Bias for layer 2 128x1
    float* OPL_TwoHost[H*BS];
    // OPL2: 128xBS
    /* Layer 3 - Reduction */
    float* weightsThreeHost[O*H];
    float* biasThreeHost[O*BS];
    float* OPL_RedHost[O*BS];
    // OPL3: 10xBS 
    /* Layer 4 - Output */
    float* oplFinalHost[O] = {0.f};  // Output layer; 10x1

    // Initialize host values
    randInit(weightsOneHost);
    randInit(biasOneHost);
    randInit(weightsTwoHost);
    randInit(biasTwoHost);
    randInit(weightsThreeHost);
    randInit(biasTwoHost); 

    // Set up device memory. cudaMalloc allocates device memory
    float *ipl, *weightsOne, *biasOne, *OPL_One;
    float *weightsTwo, *biasTwo, *OPL_Two;
    float *weightsThree, *biasThree, *OPL_Red;
    float *oplFinal;

    CHECK_CUDA(cudaMalloc(&ipl, I * BS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&weightsOne, H * I * sizeof(float)));
    CHECK_CUDA(cudaMallco(&biasOne, H * BS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&OPL_One, H * BS * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&weightsTwo, H * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&biasTwo, H * BS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&OPL_Two, H * BS * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&weightsThree, O * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&biasThree, O * BS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&OPL_Red, O * BS * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&oplFinal, O * sizeof(float)));

    // Copy intial values over-- all random, except for 
    CHECK_CUDA(cudaMemcpy(ipl, iplHost, I * BS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(weightsOne, weightsOneHost, H * I * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(biasOne, biasOneHost, H * BS * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(weightsTwo, weightsTwoHost, H * H * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(biasTwo, biasTwoHost, H * BS * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(weightsThree, weightsThreeHost, O * H * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(biasThree, biasThreeHost, O * BS * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(oplFinal, oplFinalHost, O * sizeof(float), cudaMemcpyHostToDevice));



    // ***Set up foward prop and backward prop***
    cublasHandle_t handle; 
    CHECK_CUBLAS(cublasCreate(&handle));


    // Read in data

    // Apply forward prop and backward prop


    // Use streams for batch training? 


    cublasDestroy(&handle);
}