#ifndef NUMPY_H
#define NUMPY_H

#ifndef DEFAULT_LIB
#define DEFAULT_LIB
    #include <stdint.h>
    #include <stdlib.h>
    #include <stdio.h>
#endif

typedef struct Shape {
    uint8_t row; // height
    uint8_t col; // width
} Shape; 

typedef struct Array {
    float *data; // Values of the matrix allocated in 1D array 
    Shape shape; // shape
} Array; 

Array *create2DArray(int row, int col); // Create a 2D array
Array *matrixDot(Array *a, Array *b); // Matrix Multiplication
Array *matrixAdd(Array *a, Array *b); // Matrix Sumarization
void matrixTranspose(Array *a); // Transpose the matrix

void freeArray(Array *a); // Free memory allocated for the array

void printArray(Array *a); // Print the value of array 

#endif