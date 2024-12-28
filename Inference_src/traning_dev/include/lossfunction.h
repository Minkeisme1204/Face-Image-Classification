#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H



#ifndef DEFAULT_LIB
#define DEFAULT_LIB
    #include <stdint.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <math.h>
#endif

#include "numpy.h"
#include "configs.h"
#include "activation.h"
#include "convolution.h"
#include "fullyconnected.h"

float mean_square_error(Array *predict, Array *target);
float  yolov1_error(Array *predict, Array *target);

// Function for FullyConnected with FullyConnected input
Array *d_yolov1_error_fc(FullyConnected *fc, FullyConnected *input);

// Function for Conv2D with FullyConnected input
Array *d_yolov1_error_conv2d(FullyConnected *output);
// Size output (SxS)*2*(5) = 250 value



#endif