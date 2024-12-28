#include "batchnorm.h"

BatchNorm *createBatchNorm(Conv2D *convInput, uint8_t batchSize, float gamma, float beta) {
    /*
    Input: The address of the Conv2D structure, batch size, gamma, and beta
    Output: A pointer to the created BatchNorm structure
    */
    BatchNorm *BN = (BatchNorm*)malloc(sizeof(BatchNorm));
    BN->batchSize = batchSize;
    BN->gamma = gamma;
    BN->beta = beta;


    return BN;
}

void freeBatchNorm(BatchNorm *layer) {
    /*
    Input: The address of the BatchNorm structure to be freed
    Output: None
    */
    free(layer);
}

void batcNormFeedForward(BatchNorm *BN, Conv2D *conv) {
    /*
    Input: The address of the Conv2D structure and the batch size
    Output: None
    */
    // Implement batch normalization
    // Calculate mean and variance for each channel
    // Subtract mean and divide by variance
    // Apply scale and shift parameters
    // Update the value of the Conv2D structure
    //

    
}