nv7')(x)
x = Conv2D(32, (3, 3), activation='relu', name='conv8')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool5')(x)