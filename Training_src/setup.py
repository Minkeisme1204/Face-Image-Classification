import cupy as cp

class Conv2D(object):
    def __init__(self, in_filter, filters, kernel=(3, 3), activation="relu", stride=1, padding=0, regularization=False, name=None):
        """
        Initializes a Conv2D layer.
        :param in_filter: Number of input channels.
        :param filters: Number of output filters.
        :param kernel: Tuple defining the kernel size (height, width).
        :param activation: Activation function name.
        :param stride: Stride of the convolution.
        :param padding: Padding applied to the input.
        :param regularization: Whether to use regularization.
        :param name: Optional name for the layer.
        """
        self.stride = stride
        self.padding = padding
        self.kernel = xavier_init_kernel(in_filter=in_filter, filters=filters, kernel=kernel)  # Shape (filters, in_filter, kernel_h, kernel_w)
        self.bias = xavier_init_kernel(in_filter=1, filters=filters, kernel=(1, 1))  # Shape (filters, 1)
        self.bias = self.bias.reshape(filters, 1)
        self.activation = activation
        self.regularization = regularization
        self.name = name
        self.in_filter = in_filter

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass for the Conv2D layer.
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Output tensor after convolution and activation.
        """
        batch_size, channels, height, width = x.shape
        assert channels == self.in_filter, "Input channels must match the number of filters in the kernel."

        # Padding the input
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        x_padded = cp.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

        # Calculate output dimensions
        kernel_h, kernel_w = self.kernel.shape[2], self.kernel.shape[3]
        new_h = (height + 2 * self.padding - kernel_h) // self.stride + 1
        new_w = (width + 2 * self.padding - kernel_w) // self.stride + 1

        # Stride-based slicing
        shape = (batch_size, channels, new_h, new_w, kernel_h, kernel_w)
        strides = (
            x_padded.strides[0],
            x_padded.strides[1],
            x_padded.strides[2] * self.stride,
            x_padded.strides[3] * self.stride,
            x_padded.strides[2],
            x_padded.strides[3],
        )
        sliced = cp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        sliced = sliced.reshape(batch_size, channels, new_h * new_w, kernel_h * kernel_w)

        # Perform convolution
        flatten_kernel = self.kernel.reshape(self.kernel.shape[0], self.kernel.shape[1], -1)  # Shape: (filters, in_filter, kernel_h*kernel_w)
        conv_output = cp.einsum("bfhwc,foi->bfho", sliced, flatten_kernel)  # Einstein summation for convolution

        # Add bias
        conv_output += self.bias[:, cp.newaxis, cp.newaxis]

        # Reshape output to (batch_size, filters, new_h, new_w)
        conv_output = conv_output.reshape(batch_size, self.kernel.shape[0], new_h, new_w)

        # Apply activation function
        output = f_activation[self.activation](conv_output)
        return output
