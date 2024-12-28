#include "cuda_runtime.h"
#include <stdio.h>

__global__ void conv2D_kernel(float *input, float *output, float *kernel,
    int height, int width, int channels, int filter, int kernel_size, int stride, int new_height, int new_width) {
        /*
        Input dimensions: batch_size, channel, height, width
        Output dimensions: batch_size, new_height, new_width, channels
        Kernel dimensions: kernel_size, kernel_size, channels

        Grid size: output_height, output_width, batch*filter_out
        Block dim: kernelsize, kernelsize
        Stride: stride, stride, stride
        */
        __shared__ float array[kernel_size*kernel_size];
        
        // Know which Filter is used
        int filterId = blockIdx.z % filter;

        // Know which batch are working with
        int batchId = (blockIdx.z - filterId)/filter;
        
        // 

        // Load input data


} 
int main(int argc, char **argv) {
    int batchsize = 10;
    int height = 16; 
    int width = 16; 
    int channels = 8;   
    int kernel = 3; 
    
    // Allocate memory for input and output tensors
    float *input = (float *)malloc(batchsize * height * width * channels * sizeof(float));
    for (int i = 0; i < batchsize * height * width * channels; i++) {
        input[i] = i + 1; 
        printf("%d ", (int)input[i]);
    }

    for (int i = 0; i < batchsize; i++ ) {
        printf("feature maps of batch: %d\n", i);
        for (int j = 0; j < channels; j++) {
            printf("channel: %d\n", j);
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    printf("%d ", (int)input[i * height * width * channels + j * height * width + k * height + l]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}