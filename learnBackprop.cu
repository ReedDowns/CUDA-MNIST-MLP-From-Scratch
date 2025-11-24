#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

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

// print matrix
void printmat (float *a, int d1, int d2) {
    for (int i = 0; i < d1; i++) {
        for (int j = 0; j < d2; j++) 
            printf("%8.3f", a[i * d2 + j]);
        printf("\n");
        }
    printf("\n");
}

__global__ void lossCalc(float* A, float* B, float* C, int dim1, int dim2) {
    int id = (threadIdx.x + blockDim.x*blockIdx.x) * // Rows past
             blockDim.y*gridDim.y +   // Total count in one row
             blockIdx.y * blockDim.y +  // Count columns passed til this block
             threadIdx.y; // Count cols passed within this block

    /* Idexing pitfall: Can't think about it as "find block, then find in block.". It needs to be thought of as total rows past * length of row, then total columns past on this row. Unfortunately...*/

    /* Still safe. id won't return something silly like y,x 
       instead of x,y and call it the same */

    if (id < dim1*dim2) {
        C[id] = 2.f * (A[id] - B[id]);
    }
}

// GPU 1-sample softmax gradient. Not setup for GPU though
__global__ void softmaxGrad(int sBound, float* dCdA, float* A, float* dAdz) {
    // Womp womp (pre-math)
    // Need S to be a float declared here; not a pointer passed to the function!
    float S=0.f;

    // Two steps:
    // 1. Compute the 
    for (int i = 0; i < sBound; i++) {
        S += dCdA[i]*A[i];
    }

    // 2. Apply the vector-version of the 
    for (int i = 0; i < sBound; i++) {
        dAdz[i] = A[i] * (dCdA[i] - S);
    }
}

// Non-GPU implementation
void batchedSoftmaxGrad(int sBound, float* dCdA, float* A, float* dAdz, int bs) {
    // Womp womp (pre-math)
    // (Post math) The softmax derivative (Jacobian bc non-zero off-diagonal elements),
    // then rewritten as a summation per row/col when a target is multiplied by the 
    // softmax Jacobian.
    // Per row: A_n = A_n (1 - sum(A_m)). Done once per output element n per batch. Done
    // below, also including dC/dA

    // Assuming row-major order

    for (int j = 0; j < bs; j++) {
        float S = 0.f;
        int offset = j * sBound;
        
        // Calculate sum * upstream grad
        for (int i = 0; i < sBound; i++) {
            S += dCdA[offset + i]*A[offset + i];
        }

        // Appy A_n, including upstream grad pt22
        for (int i = 0; i < sBound; i++) {
            dAdz[offset + i] = A[offset + i] * (dCdA[offset + i] - S);
        }
    }
}

// GPU implementation
__global__ void batchedSoftmaxGrad(int sBound, float* dCdA, float* A, float* dCdz, int bs) {
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;
    int id = row * bs + col;

    float S = 0.f;
    
    /* - Iterate over x
       - Need to sum over column (iterate over row) 
       - Assign x to row
       - Offset index by (row number - 1) * batch size to get next value 
       - Used math to turn "tensor" -> "matrix with a sum"
    */
    if (row < sBound && col < bs) {
        
        // Calculate sum * upstream grad
        for (int i = 0; i < sBound; i++) {
            S += dCdA[bs * i + col]*A[bs * i + col]; // Each column is a sample...
        }

        // I think this actually becomes dCdz (yes; dAdz is built into the code)
        dCdz[id] = A[id] * (dCdA[id] - S);


        // // Appy A_n, including upstream grad pt2
        // for (int i = 0; i < sBound; i++) {
        //     dAdz[id] = A[bs * i + col] * (dCdA[bs * i + col] - S);
        // }
        /* Because the for loop above is just to get to all the dAdz elements,
           it just becomes the dAdz term preceding the for loop! Basically, any
           for-loop that iterates over a result array is replaced by thread indexing.*/
    }
}

// Unbatched: getWeightGrad
__global__ void single_getWeightGrad (float* dCdz, /*dCdA * dAdz, d_out=M x 1*/
                               float* A_Lm1, /* A^L-1, d_in=I x 1*/
                               float* w,     /* W^L, M x I */
                               float* dCdw,  /* dCdw, M x I */
                               float d1, float d2, float bs /* M, I, BS*/
                               /* I, M, BS are local to this function and its 
                                  comments. Input dimensions, output dimensions,
                                  and batch size for layer L and layer L-1. */
) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int id  = row*d2 + col;

    if (row < d1 && col < d2) dCdw = dCdz[row] + A_Lm1[col];
    /* Each row of dCdw is associated with an output; the chained grad of which
       is stored in the row of dCdz. So, for each row of dCdw, dCdz[row] must be
       applied to each element in that row. Each element along a row is associated 
       with dz/dw = A^L-1. So dCdw[col] -> A^L-1[col]. Thus, 
       dCdw[row, col] = dCdz[row] * A^L-1[col].
       
       dCdz and A^L-1 both store samples per column. So, variation within one sample
       has to be done by row. Easy logic for dCdz because dCdw rows -> dCdz rows.
       
       Less intuitive for A^L-1, because dCdz cols -> A^L-1 rows.
       
       **Note: Both dCdz and A^L-1 are stored in row-major order. With col. dimension
       equal to 1, col vs row doesn't matter because one sample is stored as
       s1 = [a11, a21, .. ai1] within memory OR s1 = [a11, a12, .. a1i]. */
}

__global__ void getWeightGrad(float* dCdz, /* Contains dC/dA * dA/dz , pretty sure it's d_out x BS*/
                              float* A_Lm1, /* dz/dw = A^L-1;  */
                              float* dCdw, 
                              float* w, 
                              int d1, int d2, int bs /* d_out, d_in, batch_size */
) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int id = row * d2 + col;

    // int idxPerBatch = d1 * d2;
    // Adjustment incorrect


    if (row < d1 && col < d2) {
        // I swear I'm missing the matrix elements to this shit
        // A whole bunch of weights are gonna have the same PD-- yes
        // Each input node A^L-1 of length I will be the derivative for 
        // M weights connecting the input node a_i^L-1 to its output a_m^L.
        // Now, how the fuck do I code that? 
        // I think, 

        // For each column, ergo col + row * m | 0 < m < M = d1
        //   dCdw = dzdw * dzdw -> dCdw = g * a_col,
        // row is an output z_m. each element of the row gets a different a^L-1_i
        for (int b = 0; b < bs; b++) {
                                // + i *bs
            dCdw[id] += dCdz[row*bs + b] *  // Start at flattened idx of row, increment by b+=1
                        A_Lm1[col*bs + b];  // ALSO row-major; IxBS. col is the dCdw col!!!

            /* Both are stored in row-major order. row/col variables are just 
               a readability convention. For dCdz, each column is a sample aka dimension MxBS, 
               for A each column is a sample, dimension IxBS. 
               To ChatGPT's point, A should be A^T. 
               
               However, as mentioned, each output A^L_m has all I A_^L PDs, so each of dCdw_row,i needs
               to have that output's chain rule PD * the respective A_^L-1_i. Since dCdw is 
               of dimensions MxI, we use M=rows, MxBS on dCdz. Meaning, if I want each value
               from a row, the output values for all different samples, just push by i. 
               Whereas, A_Lm1 needs each value from a column, so memory needs to be skipped 
               by row. Lenght of one row is number of columns; in IxBS, that's BS.*/

            // // How w[id] should eventually be updated
            // w[id] -= dCdw[id] * learningRate;

        }
    }
}

// Still on layer L, get biases
__global__ void get_and_update_BiasGrad(float* dCdz, float* b, int d1, int bs) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < d1) for (int i = 0; i < bs; i++) b[row] += dCdz[row*bs + i];
}

// Get intermediate dC/dA^L-1
__global__ void getAlm1Grad(float* dCdz,    /* dCdz, O*BS */ 
                            float* w,       /* Weights, OxI */
                            float* dCdAlm1, /* Pre-batch sums, should be IxBS */
                            int d1, int d2, int bs /* O, I, BS */
) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int id  = row * bs + col; // 

    // dCdA^L-1 = ... dz^L/dA^L-1; dz^L/dA^L-1 = w^L; w^L { columns per A^L-1_i 
    // For a singles sample, dCdz, each single entry per column (each row) 
    // corresponds to to one A_i, so, sum across rows; ergo, starting row
    // through ending row of weights to get sum of PDs affecting A, each multiplied
    // by their respective output layer dC/dz^L_m
    if (row < d2 /*aka I*/ && col < bs) {

        // Sum along BS
        for (int i = 0; i < bs; i++) {
            dCdAlm1[id] += dCdz[row * bs + i] * w[col + bs * i];
        }
    }
}
// Get dC/dz^L-1; calculate dA^L-1/dz^L-1; relu for most layers
__global__ void reluGrad(float* dCdA, float* dCdz, float* z, int d1, int bs) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockDim.y * blockDim.y;
    int id =  row * bs + col;

    // Coming through the first time, it was relu(z^L). So,
    if (row < d1 && col < bs) {
        if (z > 0) dCdz = dCdA; 
        else dCdz = 0.f; 
    }
}

#define O 10
#define H
#define I 
#define BS 

int main () {

    // Step 1: Cost/Loss calculation
    float lossVector[O];
    float labelVector[O];
    float finalOPL[O];

   
    //No thread dims or device data transfer yet (dCdA)
    lossCalc<<< >>>(finalOPL, labelVector, lossVector, O, 1);

    // Step 2: Killer bees. Lots of Killer Bees
    // Returns dAdz (from final layer)
    float dCdz_L[O*BS];
    batchedSoftmaxGrad<<< >>>(O, lossVector, finalOPL, dCdz_L, BS);

    // Step 3: Get info for Layer L 
    // Need weight and bias gradients from layer L
    float OPL_Two[O*BS]; // Layer L-1, A
    float weightsThree[O*H]; // Layer L
    float dCdw_L[O*H];  // Layer L (finding)
    getWeightGrad<<< >>>(dCdz_L, OPL_Two, dCdw_L, weightsThree, O, H, BS);

    float biasThree[O*BS];
    get_and_update_BiasGrad<<< >>>(dCdz_L, biasThree, O, BS);

    // Step 4: Calculate intermediate PDs for layer L-1    
    // dC/dA^L-1 Onto the N-1 layer!
    // Need dCdz * dzd/A^L-1
    float dCdAlm1_Lm1[O*BS];
    getAlm1Grad<<< >>>(dCdz_L, weightsThree, dCdAlm1_Lm1, O, H, BS);
    
    // dC/dz^L-1, need dA/dz^L-1
    float dCdzlm1_Lm1[O*BS];
    float z3[O*BS];
    reluGrad<<< >>>(dCdzlm1_Lm1, dCdz_L, z3, O, BS);

    // Step 5: Get Weight information for Layer L-1
    float weightsTwo[H*H];
    float OPL_One[H*BS];
    float dCdw_Lm1[H*H];
    getWeightGrad<<< >>>(dCdzlm1_Lm1, OPL_One, dCdw_Lm1, weightsTwo, O, H, BS);

    float biasTwo[H*BS];
    get_and_update_BiasGrad<<< >>>(dCdzlm1_Lm1, biasTwo, O, BS);

    // Step 6: Calculate intermediate information for L-2 
    float dCdAlm2_Lm2[H*H];
    getAlm1Grad<<< >>>(dCdzlm1_Lm1, weightsTwo, dCdAlm2_Lm2, H, H, BS);

    float dCdzlm1_Lm2[H*BS];
    float z2[H*BS];
    reluGrad<<< >>>(dCdAlm2_Lm2, dCdzlm1_Lm2, z2, O, BS);

    // Step 7: Calculate weight info for L-2
    float weightsOne[H*I];
    float batchedInput[H*BS];
    float dCdwlm1_Lm2[H*H];
    getWeightGrad<<< >>>(dCdzlm1_Lm2, batchedInput, dCdwlm1_Lm2, weightsOne, H, I BS);

    float biasOne[H*BS];
    get_and_update_BiasGrad<<< >>>(dCdzlm1_Lm2, biasOne, H, BS);

    

    Fuck
    Fuck.butInBlue
    yellowFuck()
    // Green with Fucks
    


    return 0;
}


/* Note for me: pointer implementation still frequently gets me. If I wasn't so 
   methodical in having my code checked ChatGPT or Copilot, it would've burned me 
   at least three times, only one of which I saw coming beforehand.
   
   In this case, I pasesed a float to a function as a pointer, then used it as an 
   accumulator. In practice, this would've just moved the pointer along memory 
   rather than summing my quantities. */

/* Review again *exactly* how GPU kernels assign threads to tasks. In this case, why 
   would the softmax CPU kernel only be applied to one thread? 
   ? Thread management replace for loops ? 
   Whenever a for-loop iterates over the end result, which is owned by a CUDA thread,
   it's replaced by CUDA's indexing! */

/* Further gradSoftmax optimziation would be to use a triangular matrix and mirror it? */