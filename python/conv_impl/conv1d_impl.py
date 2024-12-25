import numpy as np


def conv1d(input, kernel, stride=1, padding='valid'):
    kernel_length = len(kernel)

    # Apply padding
    if padding == 'same':
        pad_width = (kernel_length - 1) // 2
        input_padded = np.pad(input, pad_width, mode='constant')
    elif padding == 'valid':
        input_padded = input
    else:
        raise ValueError("Padding must be 'valid' or 'same'")

    padded_length = len(input_padded)

    # Calculate output length
    output_length = (padded_length - kernel_length) // stride + 1
    output = np.zeros(output_length)

    # Perform convolution
    for i in range(output_length):
        start = i * stride
        end = start + kernel_length
        output[i] = np.sum(input_padded[start:end] * kernel)

    return output


# 示例
input_signal = np.array([1, 2, 3, 4, 5])
kernel = np.array([1, 0, -1])
output_signal = conv1d(input_signal, kernel, stride=1, padding='valid')
print("Output signal:", output_signal)
