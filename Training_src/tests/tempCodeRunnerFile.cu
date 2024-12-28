or (int i = 0; i < batchsize; i++ ) {
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