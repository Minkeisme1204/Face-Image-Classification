#ifndef FULLCONNECTED
#define FULLCONNECTED

#ifndef DEFAULT_LIB
#define DEFAULT_LIB
    #include <stdint.h>
    #include <stdlib.h>
    #include <stdio.h>
#endif

#ifndef MATH_H
#define MATH_H
    #include <math.h>
#endif

#ifndef NUMPY_H
// #define NUMPY_H
    #include "numpy.h"
#endif

#include "activation.h"
#include "convolution.h"
#include "initialize.h"

/*
Attributes of a Fully Connected Layer:
    output: Output of the layer
    units: Number of neurons in the layer
    weights: Weight matrix of shape (units)
    bias: Bias vector of shape (units)
    isRegularization: Flag to indicate if regularization is applied (0: No regularization, 1: L1 regularization, 2: L2 regularization)

Methods:
*/

typedef struct FullyConnected {
    // Public
    uint16_t units; 
    float (*activation)(float a);
    int8_t checkFlag; // Flag to indicate if regularization is applied
    // 11 is regularization and is output layer
    // 10 is regularization and is hidden layer
    // 00 is no regularization and is output layer
    // 01 is no regularization and is hidden layer

    // Private
    Array *output; // Output of the layer
    float *weights; // Weight matrix
    float *bias; // Bias vector

    #ifdef TRAINING
        float *h; // h = w*x + b
        Array* (*derivative)(Array *a);
        float *dE_dh;

        float *dE_dw; 
        float *dE_db;
    #endif

} FullyConnected;

FullyConnected *createFullyConnected(uint16_t units, Array* (*activation)(Array *a), int8_t isRegularization);
void freeFullyConnected(FullyConnected *fc);
void fullyConnectedFeedForward(FullyConnected *fc, FullyConnected *input);
#endif