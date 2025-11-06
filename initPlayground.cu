// Test printing random numbers!
#include <stdio.h>
#include <stdlib.h>
// #include <cuda_runtime.h>

#define M 3
#define N 2

// Choppy [0, 1] random number generator to fill + initialize matrices
void randInit(float* A, size_t n) { 
    for (size_t i = 0; i < n; i++) {
        A[i]=(float)rand()/(float)RAND_MAX;
    }
}

void printMat (float *A, int D1, int D2) {
    for (int i = 0; i < D1; i++) {
        for (int j = 0; j < D2; j++) 
            printf("%8.3f", A[i * D2 + j]);
        printf("\n");
        }
    printf("\n");
}



int main() {
    
    float TEST[M * N] = {0.0f, 1.f, 2.f, 3.f, 4.f, 5.f};
    // size_t lenTest = sizeof(TEST) / sizeof(float);

    printMat(TEST, M, N);
    randInit(TEST, M*N);
    printMat(TEST, M, N);

    // Try 0s init
    float test2[M*N] = { 0.f }; //0.f, 0.f, 0.f, 0.f, 0.f};
    printMat(test2, M, N);
    
    return 0;
}