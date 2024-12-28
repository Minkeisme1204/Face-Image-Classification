#include "numpy.h"

Array* create2DArray(int row, int col) {
    Array *arr = (Array*)malloc(sizeof(Array));
    arr->data = (float*)malloc(row * col * sizeof(float));
    arr->shape.row = row;
    arr->shape.col = col;
    return arr;
}

void freeArray(Array *a) {
    free(a->data);
    free(a);
}

void printArray(Array *a) {
    for (int i = 0; i < a->shape.row; i++) {
        for (int j = 0; j < a->shape.col; j++) {
            printf("%f ", a->data[i * a->shape.col + j]);
        }
        printf("\n");
    }
    printf("\n");  // Print newline for better readability.
}

Array *matrixDot(Array *a, Array *b) {
    if (a->shape.col != b->shape.row) { // Check condition for valid multi-dimensional
        fprintf(stderr, "Invalid Matrix size for dot operation: shape(%d, %d) and shape(%d, %d)\n", 
            a->shape.row, a->shape.col, b->shape.row, b->shape.col);
        return NULL;
    }

    // TODO: Implement matrix multiplication using vectorized operations or CUDA for better performance and scalability.
    #ifdef CUDA 
    // Block of code for matrix multiplication with CUDA supports
    //
    #endif

    Array *result = create2DArray(a->shape.row, b->shape.col); 
    for (int i = 0; i < a->shape.row; i++) {// Multiply all the rows in a with the collumns in b
        for (int j = 0; j < b->shape.col; j++) {
            for (int k = 0; k < a->shape.col; k++) {
                result->data[i * b->shape.col + j] += a->data[i * a->shape.col + k] * b->data[k * b->shape.col + j];
            }
        }
    } 
    return result;
}

void matrixTranspose(Array *a) {
    int temp; 
    for (int i = 0; i < a->shape.row; i++) {
        for (int j = 0; j < a->shape.col; j++) {
            if (i < j) {
                temp = a->data[i * a->shape.row + j];
                a->data[i * a->shape.row + j] = a->data[j * a->shape.col + i];
                a->data[j * a->shape.col + i] = temp;  // Swap values to transpose the matrix.
                // a->data[j * a->shape.col + i] = 0;
            }
        }
    }
    temp = a->shape.row; 
    a->shape.row = a->shape.col;
    a->shape.col = temp; // Update shape to reflect the transposed matrix.
}

Array *matrixAdd(Array *a, Array *b) {
    if (a->shape.row != b->shape.row || a->shape.col != b->shape.col) {// Check the condition of valid indices
        fprintf(stderr, "Invalid Matrix size for addition: shape(%d, %d) and shape(%d, %d)\n", 
            a->shape.row, a->shape.col, b->shape.row, b->shape.col);
        return NULL;
    }
    
    Array *result = create2DArray(a->shape.row, a->shape.col);
    for (int i = 0; i < a->shape.row; i++) {
        for (int j = 0; j < a->shape.col; j++) {
            result->data[i * a->shape.col + j] = a->data[i * a->shape.col + j] + b->data[i * b->shape.col + j];
        }
    }

    return result; 
}

