#include "activation.h"

float sigmoid(float a) {
    return 1 / (1 + exp(-a));
}

float relu(float a) {
    return a > 0 ? a : 0;
}

float exponential(float a) {
    return exp(a);
}