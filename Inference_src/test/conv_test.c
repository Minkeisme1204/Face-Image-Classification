#include "convolution.h"
#define UNIT_KERNEL UNIT_KERNEL

int main(int argc, char *argv[]) {
    // Create two 2D arrays of size 3x3 and fill them with random numbers.
    Shape s = {3, 3};
    Shape stride = {1, 1};
    Conv2D* a = createConv2D(2, s, stride, 2, relu, 0);
    a->outputShape.row = 5;
    a->outputShape.col = 5;
    for (int i = 0; i < a->filter; i++) a->output[i] = malloc(a->outputShape.col * a->outputShape.row * sizeof(float));

    int k = 0; 
    for (int f = 0; f < a->filter; f++) {
        for (int i = 0; i < a->outputShape.row; i++) {
            for (int j = 0; j < a->outputShape.col; j++) {
                a->output[f][i * a->outputShape.col + j] = k++;
            }
        }
    }

    for (int f = 0; f < a->filter; f++) {
        printf("Output of filter %d:\n", f);
        for (int i = 0; i < a->outputShape.row; i++) {
            for (int j = 0; j < a->outputShape.col; j++) {
                printf("%f ", a->output[f][i * a->outputShape.col + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Test Convolution computing
    // Init kernel weights
    Conv2D *conv = createConv2D(4, s, stride, 2, relu, 0);
    for (int f = 0; f < conv->filter; f++) {
        for (int i = 0; i < conv->kernelShape.row; i++) {
            for (int j = 0; j < conv->kernelShape.col; j++) {
                if (i == j) conv->kernel[f][i * conv->kernelShape.col + j] = 1; 
            }
        }
    }

    for (int f = 0; f < conv->filter; f++) {
        printf("Kernel weights of filter %d:\n", f);
        for (int i = 0; i < conv->kernelShape.row; i++) {
            for (int j = 0; j < conv->kernelShape.col; j++) {
                printf("%f ", conv->kernel[f][i * conv->kernelShape.col + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    convolutionFeedForward(conv, a);

    printf("Output after convolution\n");
    printf("Shape of output: %d, %d\n", conv->outputShape.row, conv->outputShape.col);

    for (int f = 0; f < conv->filter; f++) {
        printf("Output of filter %d:\n", f);
        for (int i = 0; i < conv->outputShape.row; i++) {
            for (int j = 0; j < conv->outputShape.col; j++) {
                printf("%f ", conv->output[f][i * conv->outputShape.col + j]); 
            }
            printf("\n");
        }
        printf("\n");
    }


    return 0;
}
    // Perform matrix