#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void readFile(const char* filename, float* samples, float* labels) {
    FILE *fPtr = fopen(filename, "r");
    if (fPtr == NULL) { // ptr at null means there's no address ergo no file.
        perror("Error opening file. Double-check the name.");
        exit(EXIT_FAILURE);
    }

    // Set a buffer for each line of the file
    char line[5120];
    int rows=0;
    bool headerPresent=true;
    while (fgets(line, sizeof(line), fPtr)) {

        if (headerPresent) {
            char *token = strtok(line, ",\r\n"); // Pointer to mem address of beginning of line?
            while (token != NULL) {
                token = strtok(NULL, ",\r\n");
            }
            headerPresent=false;
            continue;
        }


        char *token = strtok(line, ",\r\n"); // Pointer to mem address of beginning of line?

        // First token
        labels[rows] = atof(token);
        token = strtok(NULL, ",\r\n");

        // printf("Starting str loop:\n");
        int cols=0;
        while (token != NULL) {
            samples[rows*784+cols] = atof(token);
            token = strtok(NULL, ",\r\n");
            cols++;
        }
        if (rows % 10 < 1) {
            printf("Row: %d\n", rows);
        }
        rows++;
    }
    printf("closing file...\n");
    fclose(fPtr);
    printf("File closed.\n");

}


int main () {
    printf("I'm losing my mind)\n");

    float *trainSamples = (float*)malloc(60000*784*sizeof(float));
    float *trainLabels = (float*)malloc(60000*sizeof(float));
    
    printf("Do we even make it into thte funtion??\n");

    readFile("mnist_train.csv", trainSamples, trainLabels);
    // readFile("mnist_test.csv", )

    printf("Row 1: ");
    for (int i = 0; i < 784; i++) {
        printf("%f, ", trainSamples[i]);
    }

    printf("\n\n");

    printf("Label[0]: %f\n", trainLabels[0]);

    free(trainSamples);
    free(trainLabels);
    return 0;
}


/* Things I still don't yet perfectly understand from this experience:
   1) The double-pointer suggestion + function-scope malloc for sample and label arrays
   2) Robust methods for header handling, including strchr(), multiple fgets file opens and closes, et etc. 
*/