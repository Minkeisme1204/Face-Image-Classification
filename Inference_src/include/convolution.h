#ifndef CONV2D
#define CONV2D

#include "activation.h"

// Components of CONV2D: 
/*
Value Matrix 
Kernel
Padding 
Stride
Output Matrix
activation function
L1, L2 regularization
*/

typedef Array Kernel;
typedef Shape Stride; 

typedef struct Conv2D {
    uint8_t filter;
    Stride stride; 
    int8_t padding;
    float (*activation)(float a);
    int8_t isRegularization;

    float **output; // output of the previous layer  
    float **kernel; 
    float *bias;   

    Shape outputShape; 
    Shape kernelShape;

    #ifdef TRAINING
        Array* (*derivative)(Array *a);
        float *dE_dh;

        float **dE_dw; 
        float *dE_db;
    #endif
} Conv2D;

Conv2D *createConv2D(uint8_t filter, Shape kernelShape, Stride stride, uint8_t padding, float (*activation)(float a), int8_t isRegularization); 
void initKernel(Kernel *kernel);
void freeConv2D(Conv2D *conv);
void convolutionFeedForward(Conv2D *conv, Conv2D *input);
void addPadding(Conv2D *conv, int padding);

#endif