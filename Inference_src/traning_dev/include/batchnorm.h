// Havent test this lib 
#ifndef BATCHNORM
#define BATCHNORM

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

#ifndef CONV2D
#include "convolution.h"
#endif

typedef struct BatchNorm {
    uint8_t batchSize;
    float gamma; 
    float beta;
    
} BatchNorm;


BatchNorm *createBatchNorm(Conv2D *convInput, uint8_t batchSize, float gamma, float beta);
void freeBatchNorm(BatchNorm * layer);
void batcNormFeedForward(BatchNorm *BN, Conv2D *conv);

#endif


