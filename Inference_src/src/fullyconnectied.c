
#include "fullyconnected.h"

FullyConnected *createFullyConnected(uint8_t units, Array* (*activation)(Array *a), int8_t checkFlag) {
    /*
    Input: number of units, activation function, and isRegularization flag
    Output: A pointer to the created FullyConnected layer
    */
    FullyConnected *fc = (FullyConnected*)malloc(sizeof(FullyConnected));
    fc->units = units;
    fc->activation = activation;
    fc->checkFlag = checkFlag;
    fc->bias = create2DArray(units, 1);

    #ifdef TRAINING
        if (fc->activation == relu) fc->derivative = d_relu;
        else if (fc->activation == sigmoid) fc->derivative = d_sigmoid;
        else if (fc->activation == softmax) fc->derivative = d_softmax;
    #endif

    return fc;
}
void freeFullyConnected(FullyConnected *fc) {
    /*
    Input: The address of the FullyConnected layer to be freed
    Output: None
    */
    freeArray(fc->weights);
    freeArray(fc->bias);
    free(fc);
}
void fullyConnectedFeedForward(FullyConnected *fc, FullyConnected *input) {
    /*
    Input: The address of the FullyConnected layer and the address of its input
    Output: None
    */
    // Perform matrix multiplication between the input and weights
    // Apply the activation function to the result
    // Update the value of the FullyConnected layer
    if (fc->weights == NULL) { // Initialize the weights
        fc->weights = create2DArray(fc->units, input->units);
    }

    fc->output = matrixDot(input->output, fc->weights);
    fc->output = matrixAdd(fc->output, fc->bias);
    for (int i = 0; i < fc->output->shape.col*fc->output->shape.row; i++) {
        fc->output->data[i] = fc->activation(fc->output->data[i]);
    }
}

#ifdef TRAINING
    float* fullyConnectedBackpropagate(FullyConnected *fc, FullyConnected *input) {
        /*
        Input: The address of the FullyConnected layer and the address of its input
        Output: None
        */
        // Compute the derivative of the loss with respect to the output
        // Apply the derivative of the activation function to the result
        // Compute the derivative of the loss with respect to the weights and bias
        // Update the value of the FullyConnected layer
        // Update the value of the input
        // Apply regularization if required

        if (fc->checkFlag & 1 == 1) {// Is output Layer
            
        }                
    }
#endif

