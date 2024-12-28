from nn_custom.Models import *

if __name__ == '__main__':
    model = Model(num_classes=6, input_shape=(128, 128, 3))
    model.add(Conv2D(in_filter=3, filters=8, kernel=(3, 3), activation='relu', stride=1, padding=1, name='conv1'))
    model.add(Conv2D(in_filter=8, filters=16, kernel=(3, 3),activation='relu', stride=2, padding=0, name='conv2'))
    model.add(Conv2D(in_filter=16, filters=16, kernel=(3, 3),activation='relu', stride=1, padding=0, name='conv3'))
    model.add(Conv2D(in_filter=16, filters=32, kernel=(3, 3),activation='relu', stride=1, padding=0, name='conv4'))
    model.add(Flatten())
    model.add(FullyConnected(in_features=59*59*32, units=64, activation='sigmoid', name='fc1'))
    model.add(FullyConnected(in_features=64, units=128, activation='sigmoid', name='fc2'))
    model.add(FullyConnected(in_features=128, units=6, activation='softmax', name='fc3'))

    model.compile(optimizer=SGD(lr=0.01, momentum=0.99), loss_function=CatergoricalLossEntropy())