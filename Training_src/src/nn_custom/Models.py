from src.nn_custom.activations import *
from src.nn_custom.fully_connected import *
from src.nn_custom.optimizer import *
from src.nn_custom.loss_functions import *
from src.nn_custom.convolution2D import *
from src.nn_custom.maxpooling2D import *
from src.nn_custom.flatten import *

class Model(object):
    def __init__ (self, num_classes=6, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = []
        self.loss_function = None

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss):
        dL_da = self.loss_function.derivative()
        for layer in reversed(self.layers):
            dL_dx = layer.backward(dL_da)
            dL_da = dL_dx
        return dL_dx

    def compile(self, optimizer, loss_function):
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train_step(self, x_train, y_train):
        y_pred =  y_pred = self.forward(x_train)

        loss = self.loss_function(y_pred, y_train)

        dL_da = self.loss_function.derivative(y_pred, y_pred)

        self.backward(dL_da)

        self.optimizer.update(self.layers.weights, self.layers.dL_dw)
        self.optimizer.update(self.layers.bias, self.layers.dL_db)

        return loss

    def fit(self, x, y, val=None, epochs=200, batch_size=32, metrics=['accuracy'], verbose=True):
        num_samples = x.shape[0]
        self.metrics_name = metrics
        for epoch in range(1, epochs + 1):
            # Shuffle the data at the beginning of each epoch
            permutation = cp.random.permutation(num_samples)
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]

            self.epoch_loss = 0.0
            self.epoch_metrics = cp.zeros(shape=metrics.shape[0])
            num_batches = int(cp.ceil(num_samples / batch_size))

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_samples)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Perform a training step and accumulate loss
                loss = self.train_step(x_batch, y_batch)
                self.epoch_loss += loss

                # Compute accuracy for the batch
                y_pred = self.forward(x_batch)

                # Compute metrics for the batch
                for i, metric in enumerate(metrics):
                    temp = metric[i](y_pred, y_batch)
                    self.epoch_metrics[i] += temp

            # Compute average loss and accuracy over the epoch
            self.avg_loss = self.epoch_loss / num_batches
            self.avg_metric = self.epoch_metrics / num_batches
    
