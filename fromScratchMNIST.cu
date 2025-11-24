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

// print matrix
void printmat (float *a, int d1, int d2) {
    for (int i = 0; i < d1; i++) {
        for (int j = 0; j < d2; j++) 
            printf("%8.3f", a[i * d2 + j]);
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

__global__ void add_and_activate(
    float* m1,  // Matrix 1, bias
    float* m2,  // Matrix 2, z, Weights x prevLayer + bias
    float* m3,  // Matrix 3, output, activated z
    int n,      // Number of weights (H usually, ergo rows)
    int b,      // Batch size (BS usually, ergo columns)
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

// One layer; input -> weights -> bias -> activation.
// Outputs stored in output
// Notes are for geneeralized hidden layer
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
    float alpha = 1.f;
    float beta  = 0.f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D3, D1, D2,
                             &alpha, input, D3, weights, D2, 
                             &beta, output, D3));

    /* This layer now stored in output (external scope)*/

    // Fused bias and activation. Values stored in output.
    add_and_activate<<<gridDim, blockDim>>>(bias, output, D2, D1, activation);
    
    /* Whatever array passed to output is now activated and ready for next layer.
       Shape is D1 x D3 (H x BS for most layers.)*/
}

void readFile(const char* filename, float* samples, float* labels) {
    // Open file
    FILE *fPtr = fopen(filename, "r");
    if (fPtr == NULL) { // ptr at null means there's no address ergo no file.
        perror("Error opening file. Double-check the name.");
        exit(EXIT_FAILURE);
    }

    // Set a buffer for each line of the file
    char line[5120]; // NOT robust if lines are longer than buffer
    int rows=0;
    bool headerPresent=true;
    while (fgets(line, sizeof(line), fPtr)) {

        // Single call to get rid of header
        if (headerPresent) {
            char *token = strtok(line, ",\r\n"); // Pointer to mem address of beginning of line?
            while (token != NULL) {
                token = strtok(NULL, ",\r\n");
            }
            headerPresent=false;
            continue;
        }

        // Start token chain
        char *token = strtok(line, ",\r\n"); // Pointer to mem address of beginning of line?

        // First token per row to label
        labels[rows] = atof(token);
        token = strtok(NULL, ",\r\n");

        // Further tokens per row to sample
        int cols=0;
        while (token != NULL) {
            samples[rows*784+cols] = atof(token);
            token = strtok(NULL, ",\r\n");
            cols++;
        }
        rows++;
    }
    fclose(fPtr);
}



void main() {
    // Hyperparameters?


    // Initialize arrays on host (before passing to device).
    /* Layer 0 - Input*/
    float iplHost[I*BS];          // Input layer, 784x1  A^L-3
    /* Layer 1 - Hidden 1 */
    float weightsOneHost[H*I];     // Weights for layer one 128x784
    float biasOneHost[H*BS];          // Bias for layer 128x1
    float z1Host[H*BS];         // save partial step for backprop
    float OPL_OneHost[H*BS];    // A^L-2
    // OPL1: 128xBS
    /* Layer 2 - Hidden 2 */
    float weightsTwoHost[H*H];     // Weights for layer two 128x128
    float biasTwoHost[H*BS];          // Bias for layer 2 128x1
    float z2Host[H*BS];         // save partial step for backprop
    float OPL_TwoHost[H*BS];       // A^L-1
    // OPL2: 128xBS
    /* Layer 3 - Reduction */
    float weightsThreeHost[O*H]; // W^L-1
    float biasThreeHost[O*BS];   // b^L-1
    float z3Host[O*BS];          // save partial step for backprop
    float OPL_RedHost[O*BS];  // A^L
    // OPL3: 10xBS 
    /* Layer 4 - Output */
    float labelsHost[O*BS] = {0.f};  // Output layer; 10x1; y

    // Initialize host values
    randInit(weightsOneHost, H*I);
    randInit(biasOneHost, H*BS);
    randInit(weightsTwoHost, H*H);
    randInit(biasTwoHost, H*BS);
    randInit(weightsThreeHost, O*H);
    randInit(biasThreeHost, O*BS); 

    // Set up device memory. cudaMalloc allocates device memory
    float *ipl, *weightsOne, *biasOne, *OPL_One;
    float *weightsTwo, *biasTwo, *OPL_Two;
    float *weightsThree, *biasThree, *OPL_Red;
    float *labels;

    CHECK_CUDA(cudaMalloc(&ipl, I * BS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&weightsOne, H * I * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&biasOne, H * BS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&OPL_One, H * BS * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&weightsTwo, H * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&biasTwo, H * BS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&OPL_Two, H * BS * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&weightsThree, O * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&biasThree, O * BS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&OPL_Red, O * BS * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&labels, O * sizeof(float)));

    // Copy intial values over-- all random, except for 
    CHECK_CUDA(cudaMemcpy(weightsOne, weightsOneHost, H * I * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(biasOne, biasOneHost, H * BS * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(weightsTwo, weightsTwoHost, H * H * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(biasTwo, biasTwoHost, H * BS * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(weightsThree, weightsThreeHost, O * H * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(biasThree, biasThreeHost, O * BS * sizeof(float), cudaMemcpyHostToDevice));

    // CHECK_CUDA(cudaMemcpy(oplFinal, oplFinalHost, O * sizeof(float), cudaMemcpyHostToDevice));



    // ***Set up foward prop and backward prop***
    cublasHandle_t handle; 
    CHECK_CUBLAS(cublasCreate(&handle));


    // Read in data; get train_samples and train_labels
    // Load in data: First row = header, first column = labels
    float *trainSamples = float*


    // int train_samples
    int tTlength = (int)*sizeof(train_samples) / (int)*sizeof(int) / 784; //
    int batches  = tTlength / BS;  // 2000 = 60,000 / 30
    int batchJump = BS * 784; // Num elements to jump = 

    /*For batch size, start with 30 !* Not considering how this'll 
    utilize device resources (optimally or not); going with it because 
    it's a simple number that goes into 60k evenly while maintaining 
    a reasonable quantity of passes, 2k.*/

    /* Looking at the quantity of threads per batch: 23,520 = 30 * 784. 
    18.375 = 23520/1280. So, in a perferct world, each thread handles 19 elements.
    Want block sizes to be multiples of 32, want to utilize 10 blocks for maximum 
    device compute power. So, 2352.0 threads per block: 2352 / 32 = 73.5. Closest 
    I'm going to get is 64x64 for now, executing batches one at a time.*/
    dim3 gridDim1(10);
    dim3 blockDim1(64, 64);

    /* Now, 256 x 30. 7680 threads, */
    dim3 gridDim2(10);
    dim3 blockDim2(64, 16);

    /* Now, 10 x 30  */
    dim3 gridDim3(10);
    dim3 blockDim3(32);
     
    // For each batch
    for (int iter = 0; iter < batches; iter++) {

        //!! Still need to reset matrices between each batch; maybe just move entire intialization block down here...
        
        // Set iplHost equal to this batch of training data
        int counter = 0;
        for (int j = iter*batchJump; j < (iter + 1) * batchJump; j++){
            iplHost[counter] = train_samples[j];
        }

        // Copy this training batch to device
        CHECK_CUDA(cudaMemcpy(ipl, iplHost, I * BS * sizeof(float), cudaMemcpyHostToDevice));

        // IPL -> HL1
        layer(handle, weightsOne, ipl, biasOne, OPL_One, H, I, BS, 'r', gridDim1, blockDim1);
        // HL1 -> HL2
        layer(handle, weightsTwo, OPL_One, biasTwo, OPL_Two, H, H, BS, 'r', gridDim2, blockDim2);
        // HL2 -> Reduction
        layer(handle, weightsThree, OPL_Two, biasTwo, OPL_Red, O, H, BS, 's', gridDim3, blockDim3);

        // Don't actually need an average done here because we're about to start a backpass. Just need labels, of shape O x BS to calculate the gradient (I think). So, oplFinal is obsolete. 

    }
    // Use streams for batch training? 


    cublasDestroy(&handle);
    return 0;
}