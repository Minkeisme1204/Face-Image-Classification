import cupy as cp

# Kernel and parameters
kernel = cp.ones(shape=(5, 3, 3))  # Shape of the kernel (filter, h, w)
padding = 2
stride = 1
channels_in = 2  # Number of input channels

# Create an input image (channels=2, h=5, w=5)
a = cp.arange(1, 5 * 5 * channels_in + 1).reshape(channels_in, 5, 5)  # Shape (channels=2, h=5, w=5)

# Pad the height and width only (not channels)
pad_width = ((0, 0), (padding, padding), (padding, padding))  # Only pad h, w
a_padded = cp.pad(a, pad_width, mode='constant', constant_values=0)
print(a_padded)

# Calculate new height and width
input_h, input_w = a_padded.shape[1:]  # Extract h, w from (channels, h, w)
filter, kernel_h, kernel_w = kernel.shape
new_h = (input_h - kernel_h) // stride + 1
new_w = (input_w - kernel_w) // stride + 1

# Stride calculation
shape = (a_padded.shape[0], new_h, new_w, kernel_h, kernel_w)  # Shape for (channels, new_h, new_w, h, w)
strides = (a_padded.strides[0], a_padded.strides[1] * stride, a_padded.strides[2] * stride, a_padded.strides[1], a_padded.strides[2])

# Create the sliding window view
sliced = cp.lib.stride_tricks.as_strided(a_padded, shape=shape, strides=strides)

# Reshape the slices for easier flattening (channels, new_h, new_w, h, w) -> (channels, new_h * new_w, h * w)
sliced_reshaped = sliced.reshape(a_padded.shape[0], new_h * new_w, kernel_h * kernel_w)

# Final shape to match (2, 9, 9)
sliced = sliced_reshaped

print(f"Shape of sliced matrix: {sliced.shape}")
print(sliced)
