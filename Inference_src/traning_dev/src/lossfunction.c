#define TRAINING
#include "lossfunction.h"


float mean_square_error(Array *predict, Array *target) {
    float loss = 0.0;
    for (int i = 0; i < predict->shape.row; i++) {
        for (int j = 0; j < predict->shape.col; j++) {
            loss += (predict->data[i * predict->shape.col + j] - target->data[i * target->shape.col + j]) 
                    *(predict->data[i * predict->shape.col + j] - target->data[i * target->shape.col + j]);
        }
    }
    return loss / (predict->shape.row * predict->shape.col);
}

float calculateIOU(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2) {
    float xA = fmax(x1 - w1/2., x2 - w2/2.);
    float yA = fmax(y1 - h1/2., y2 - h2/2.);
    float xB = fmin(x1 + w1/2., x2 + w2/2.);
    float yB = fmin(y1 + h1/2., y2 + h2/2.);

    float intersectionArea = fmax(0, xB - xA) * fmax(0, yB - yA);
    float unionArea = w1 * h1 + w2 * h2 - intersectionArea;

    return (float)intersectionArea / unionArea;
}

float yolov1_error(FullyConnected *predict, Array *target) {
    float total_loss = 0.0;
    /*
    Output dimension of the model is: 
        (S*S) grid for (i, j) => true id of a cell i*S + j. ~X axis
        (B = 2) number of bounding boxes. ~Y axis
        (x, y, w, h, C) parameter of a bounding box. ~Z axis.
        Id of the x is 
    */

    // Confidence loss 
    for (int i = 0; i < predict->output->shape.row; i++) {
        for (int j = 0; j < predict->output->shape.col; j++) {

            // Check if there is object in the CELL
            // Confidence Score == 1 // Every cell containing the center of gtbox is read as confidence: 1 ELSE 0 
            if (target->data[(5*2*(predict->output->shape.row*i + j)) + 5*0 + 4] == 1) {

                // Find best IOU 
                float bestIOU = 0; 
                unsigned char bestBox = 0; 
                for (int b = 0; b < 2; b++) {

                    // Calculate the IOU 
                    unsigned char id_x = (5*2*(predict->output->shape.row*i + j)) + 5*b + 0; 
                    unsigned char id_y = (5*2*(predict->output->shape.row*i + j)) + 5*b + 1;
                    unsigned char id_w = (5*2*(predict->output->shape.row*i + j)) + 5*b + 2;
                    unsigned char id_h = (5*2*(predict->output->shape.row*i + j)) + 5*b + 3; 
                    float IOU = calculateIOU(predict->output->data[id_x], predict->output->data[id_y], predict->output->data[id_w], predict->output->data[id_h], 
                                    target->data[id_x], target->data[id_y], target->data[id_w], target->data[id_h]);
                    
                    if (IOU > bestBox) {
                        bestIOU = IOU;
                        bestBox = b;
                    }
                    
                    unsigned char id_C = (5*2*(predict->output->shape.row*i + j)) + 5*bestBox + 4;

                    // Calculate Confidence Loss
                    total_loss += (predict->output->data[id_C] - target->data[id_C]) * (predict->output->data[id_C] - target->data[id_C]);

                    // Calculate Localization Loss]
                    float temp1 = sqrt(predict->output->data[id_C]) - sqrt(target->data[id_C]);
                    float temp2 = sqrt(predict->output->data[id_C]) - sqrt(target->data[id_C]);

                    total_loss += LAMDA_COORD*((predict->output->data[id_x] - target->data[id_x]) * (predict->output->data[id_x] - target->data[id_x]) 
                                                // (x^2 - x'^2)
                                                + (predict->output->data[id_y] - target->data[id_y]) * (predict->output->data[id_y] - target->data[id_y]))
                                                // (y^2 - y'^2)
                                + LAMDA_COORD*(temp1 * temp1 + temp2 * temp2);
                                                // (sqrt(w) - sqrt(w'))^2 + (sqrt(h) - sqrt(h'))^2 

                    
                }
            }
            else {
                for (int b = 0; b < 2; b++) {
                    total_loss += LAMDA_NOOBJ*predict->output->data[(5*2*(predict->output->shape.row*i + j)) + 5*b + 4];
                }
            }
        }
    }
    return total_loss;
}
