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

#define H 3
#define I 6
#define BS 3

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
    float* m2,  // Matrix 2
    int n,      // Dimension one (weighted results, ergo rows)
    int b       // Dimension two (batch size, ergo columns)
) {
    // Assign y to row, x to column. Memory stored along y. Warps along x.
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Prev. Block Rows Passsed + This block rows passed
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // %%
    int id = row * b + col;                           // Rows passed * cols in a row + col passed in this row

    /* Assigned y to row and x to columns so iteration will be within the same 
       warp AND along contiguous memory. C++ stores values along rows 
       (so, by varying columns), and warps progress along threadIdx.x. With this 
       convention, we vary x along contiguous memory and along the warps. -> False.
         Iterate along x down columns. So, along warps but not along contiguous memory.
       2) Will want to assign grid/block/thread dimensions based on input array 
       dimensions, and with consideration to batching. Each column is one sample.
       3) Will eventually want to transpose for corrected memory access.*/ 
       
    /* Each thread will own a result. That thread needs to find its column, sum its column, 
       and replace its value with the exp(val)/sum */
    if (row <  n && col < b) {

        float sumExp=0.f;
        float colMax = 0.f;
        
        // Find row max
        for (int i = 0; i < n; i++) if (m2[i * b + col] > colMax) colMax = m2[i * b + col];
        for (int i = 0; i < n; i++) {
            sumExp += expf(m2[i * b + col] - colMax); // Stride through mems by row length to hit columns
        }
        m2[id] = expf(m2[id] - colMax) / sumExp;
    }
}

__global__ void add_and_activate(
    float* m1,  // Matrix 1, bias
    float* m2,  // Matrix 2, output
    int n,      // Number of weights (H, ergo rows)
    int b,      // Batch size (BS, ergo columns)
    char activation
) {
    // Assign y to row, x to column to iterate along warps for softmax
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int id = row * b + col; 


    if (row < n && col < b) {
        // 1) Add bias
        m2[id] = m2[id] + m1[id];

        // 2) Activate the layer
        // 2a) relu
        if (activation == 'r') m2[id] = fmaxf(m2[id], 0.f);
        // 2b) softmax
        else if (activation == 's') {
            float sumExp=0.f;
            float colMax=0.f;
            for (int i=0; i<n; i++) colMax = fmaxf(m2[i*b+col],colMax); 
            for (int i=0; i<n; i++) sumExp += expf(m2[i*b+col]-colMax);
            m2[id] = expf(m2[id] - colMax) / sumExp;
        }
        // 2c) Error-- no activation; no action taken yet though
        else {
            printf("Activation required in layer func add_and_activate"); // %s:%d. Exiting... \n", __FILE__, __LINE__); 
            // exit(EXIT_FAILURE); // Apparently doesn't work on device code...
        }
    }
    /* Some notes: softmax is very non-optimized; it doesn't stride along contiguous memory. 
       See __global__ void softmax() in layerOutput.cu for more details on softmax and indexing.*/
}

void layer (
    cublasHandle_t handle,
    float* weights, /* H x I  */
    float* input,   /* I x BS */
    float* bias,    /* H x BS */
    float* output,  /* H x BS */
    int D1, int D2, int D3,  /* Rows, shared, columns. M x K, K x N. */
    char activation,
    dim3 gridDim, dim3 blockDim
) {
    // Sgemm; adjust for cuBLAS's column-major storage
    float alpha = 1.0;
    float beta  = 0.0;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D3, D1, D2,
                             &alpha, input, D3, weights, D2, 
                             &beta, output, D3));

    /* This layer now stored in output (external scope)*/

    // Fused bias and activation. Values stored in output.
    add_and_activate<<<gridDim, blockDim>>>(bias, output, D2, D1, activation);
    
    /* Whatever array passed to output is now activated and ready for next layer.
       Shape is D1 x D3 (H x BS for most layers.)*/

}


int main () {
    // Initialize matrices
    float Wei[H * I] = {0.0f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f};
    printf("Original matrix A:\n");
    printMat(Wei, H, I);

    float In[I*BS] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f};
    printf("Original matrix B:\n");
    printMat(In, I, BS);

    float Bi[H*BS] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    printf("Original matrix C:\n");
    printMat(Bi, H, BS);

    float Out[H * BS] = {0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    printf("Original matrix A:\n");
    printMat(Out, H, BS);

    // Initialize device memory
    float *dWei;
    CHECK_CUDA(cudaMalloc(&dWei, sizeof(float) * H * BS));
    CHECK_CUDA(cudaMemcpy(dWei, Wei, sizeof(float) * H * BS, cudaMemcpyHostToDevice));

    float *dIn;
    CHECK_CUDA(cudaMalloc(&dIn, sizeof(float) * I * BS));
    CHECK_CUDA(cudaMemcpy(dIn, In, sizeof(float) * I * BS, cudaMemcpyHostToDevice));

    float *dBi;
    CHECK_CUDA(cudaMalloc(&dBi, sizeof(float) * H * BS));
    CHECK_CUDA(cudaMemcpy(dBi, Bi, sizeof(float) * H * BS, cudaMemcpyHostToDevice));

    float *dOut;
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float) * H * BS));
    CHECK_CUDA(cudaMemcpy(dOut, Out, sizeof(float) * H * BS, cudaMemcpyHostToDevice));

    // Prepare device settings
    dim3 blockDim(32, 32); 
    dim3 gridDim(1);
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));


    // Layer here...     
    layer(handle, dWei, dIn, dBi, dOut, H, BS, I, 'r', gridDim, blockDim);

    // Forgot what this doees...
    cudaDeviceSynchronize();

    // Copy result back to host, print result
    CHECK_CUDA(cudaMemcpy(Out, dOut, sizeof(float) * H * BS, cudaMemcpyDeviceToHost));
    printf("Result matrix C (from GPU):\n");
    printMat(Out, H, BS);
    
    CHECK_CUDA(cudaFree(dWei));   
    CHECK_CUDA(cudaFree(dIn));   
    CHECK_CUDA(cudaFree(dBi));   
    CHECK_CUDA(cudaFree(dOut));   
    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}