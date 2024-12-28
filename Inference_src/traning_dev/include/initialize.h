#ifndef INIT 
#define INIT
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "numpy.h"

void xavier_uniform_initialize(Array *array, int input_units, int output_units) {
    float temp = 6/(input_units + output_units);
    temp = sqrt(temp);

    srand((unsigned int)time(NULL)); 

    int n = array->shape.col*array->shape.row;
    for (int i = 0; i < n; i++) {
        array->data[i] = (float)(-temp) + ((float)rand()/RAND_MAX) * (temp + temp);
    }
}

#endif