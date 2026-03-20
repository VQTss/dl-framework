import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.random.randn(out_channels)
        
        self.output_shape = (out_channels, in_channels, kernel_size, kernel_size)
    
    def forward(self, x):
        self.input = x
        self.output = np.zeros(self.output_shape)
        
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    for l in range(self.output_shape[3]):
                        self.output[i, j, k, l] = np.sum(self.input[i, j, k, l] * self.weights[i, j, k, l]) + self.bias[i]
        
        return self.output
    
    