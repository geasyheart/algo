import numpy as np


def conv2d(input, kernel, stride=(1, 1), padding='valid'):
    kernel_height, kernel_width = kernel.shape

    if padding == 'same':
        pad_height = (kernel_height - 1) // 2
        pad_width = (kernel_width - 1) // 2
        input_padded = np.pad(input, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    elif padding == 'valid':
        input_padded = input
    else:
        raise ValueError("Padding must be 'valid' or 'same'")

    padded_height, padded_width = input_padded.shape

    output_height = (padded_height - kernel_height) // stride[0] + 1
    output_width = (padded_width - kernel_width) // stride[1] + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            start_i = i * stride[0]
            start_j = j * stride[1]
            end_i = start_i + kernel_height
            end_j = start_j + kernel_width
            output[i, j] = np.sum(input_padded[start_i:end_i, start_j:end_j] * kernel)

    return output


# 示例
input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# 定义卷积核 (3x3)
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

output_matrix = conv2d(input_matrix, kernel, stride=(1, 1), padding='same')
print("Output matrix:\n", output_matrix)
