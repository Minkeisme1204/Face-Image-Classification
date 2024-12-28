#include "numpy.h"

int main(int argc, char *argv[]) {
    // Create two 2D arrays of size 3x3 and fill them with random numbers.
    Array *a = create2DArray(3, 3);
    Array *b = create2DArray(3, 3);

    float x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < a->shape.row; i++) {
        for (int j = 0; j < a->shape.col; j++) {
            a->data[i * a->shape.col + j] = (float)rand() / RAND_MAX;
            b->data[i * b->shape.col + j] = (float)rand() / RAND_MAX;
        }
    }

    // Print the original arrays.
    printf("Matrix A:\n");
    a->data = x; 
    printArray(a);

    printf("Matrix B:\n");
    printArray(b);

    Array *c = matrixDot(a, b);
    printArray(c);

    // Transpose matrix A.
    printf("Matrix A Transposed:\n");
    matrixTranspose(a);
    printArray(a);

    // freeArray(a);
    // freeArray(b);
    freeArray(c);

    c = matrixAdd(a, b);
    printArray(c);

    return 0;
}
    // Perform matrix