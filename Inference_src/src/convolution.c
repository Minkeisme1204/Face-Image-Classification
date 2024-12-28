#include "convolution.h"
#include "activation.h"

Conv2D *createConv2D(uint8_t filter, Shape kernelShape, Stride stride, uint8_t padding, float (*activation)(float a), int8_t isRegularization) {
    /*
    Input: number of filters, kernel shape, stride, padding, activation function, and isRegularization flag
    Output: A pointer to the created Conv2D structure
    */
    Conv2D *conv = (Conv2D*)malloc(sizeof(Conv2D));
    conv->filter = filter;
    conv->stride.col = stride.col;
    conv->stride.row = stride.row;
    conv->padding = padding;
    conv->activation = activation;
    conv->isRegularization = isRegularization;
    conv->kernel = (float**)malloc(filter * sizeof(float*));
    conv->kernelShape.col = kernelShape.col;
    conv->kernelShape.row = kernelShape.row;
    for (uint8_t i = 0; i < filter; i++) {
        conv->kernel[i] = malloc(conv->kernelShape.col*kernelShape.row*sizeof(float));
    }
    conv->output = (float**)malloc(filter * sizeof(float*));
    conv->bias = (float*)malloc(filter * sizeof(float));
    return conv;
}

void freeConv2D(Conv2D *conv) {
    /*
    Input: The address of the Conv2D structure to be freed
    Output: None
    */
    for (int i = 0; i < conv->filter; i++) {
        free(conv->kernel[i]);
        free(conv->output[i]);

    }
    free(conv->kernel);
    free(conv->output);
    free(conv->bias);
    free(conv);
}

void addPadding(Conv2D *conv, int padding) {
    /*
    Input: Convolutional layer, padding size
    Output: None
    */

    Shape padded_size = {conv->outputShape.row + 2*padding, conv->outputShape.col + 2*padding};

    float** padded_output = (float**)malloc(sizeof(float*) * conv->filter);

    for (uint8_t f = 0; f < conv->filter; f++) {
        padded_output[f] = (float*)malloc(padded_size.row * padded_size.col * sizeof(float));
        for (uint8_t i = 0; i < conv->outputShape.row; i++) {
            for (uint8_t j = 0; j < conv->outputShape.col; j++) {
                padded_output[f][(i + padding) * padded_size.col + (j + padding)] = conv->output[f][i * conv->outputShape.col + j];
            }
        }
    } 

    for (uint8_t i = 0; i < conv->filter;i++) {
        free(conv->output[i]);
    }
    free(conv->output);
    conv->output = padded_output;
    conv->outputShape.row = padded_size.row;
    conv->outputShape.col = padded_size.col;
    
}

void convolutionFeedForward(Conv2D *conv, Conv2D *input) {
    /*
    Input: Convolutional layer, activation function, kernel, bias, stride, and padding
    Output: A pointer to the computed output
    */

    // Calculate the Output Size of this layer 
    conv->outputShape.row = (input->outputShape.row + 2*conv->padding - conv->kernelShape.row)/conv->stride.row + 1;
    conv->outputShape.col = (input->outputShape.col + 2*conv->padding - conv->kernelShape.col)/conv->stride.col + 1;

    
    if (conv->padding > 0) addPadding(conv, conv->padding);

    #ifndef VERIFIED
        for (int f = 0; f < input->filter; f++) {
            printf("Output of filter %d:\n", f);
            for (int i = 0; i < input->outputShape.row; i++) {
                for (int j = 0; j < input->outputShape.col; j++) {
                    printf("%f ", input->output[f][i * input->outputShape.col + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    #endif

    for (uint8_t f = 0; f < conv->filter; f++) {
        conv->output[f] = (float*)malloc(conv->outputShape.row * conv->outputShape.col * sizeof(float));
        for (uint8_t i = 0; i < conv->outputShape.row; i++) {
            for (uint8_t j = 0; j < conv->outputShape.col; j++) {
                float sum = 0;
                for (uint8_t inp_f = 0; inp_f < input->filter; inp_f++) {
                    for (uint8_t k_i = 0; k_i < conv->kernelShape.row; k_i++) {
                        for (uint8_t k_j = 0; k_j < conv->kernelShape.col; k_j++) {
                            if ((i * conv->stride.row + k_i < input->outputShape.row) && (i * conv->stride.row + k_i < input->outputShape.col)) {
                                sum += input->output[inp_f][(i * conv->stride.col + k_i) * input->outputShape.col + (j * conv->stride.row + k_j)] * conv->kernel[f][k_i * conv->kernelShape.col + k_j];
                            }
                        }
                    }
                }
                conv->output[f][i * conv->outputShape.col + j] = conv->activation(sum + conv->bias[f]);
            }
        }
    }


}


