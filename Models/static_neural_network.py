import numpy as np
import torch 
import torch.nn as nn

class StaticNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes = [128, 64]) -> None:
        super(StaticNN, self).__init__()

        self.layers = []
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(self.layer_sizes)-1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))

        self.activation = nn.Tanh()
        self.ABCD = torch.Tensor(input_size, output_size, 5)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        return self.layers[-1](x)
    
    def get_layer_n_weights(self, layer):
        return layer.weight.shape[0] * layer.weight.shape[1]

    def get_n_weights(self):
        n_weights = 0
        for layer in self.layers:
            n_weights += self.get_layer_n_weights(layer)
        return n_weights

    def update_weights(self, weights):
        a = 0
        for i, layer in enumerate(self.layers):
            b = a + self.get_layer_n_weights(layer)
            w = torch.reshape(weights[a:b], (layer.weight.shape))
            self.layers[i].weight = nn.Parameter(w)
            a = b

    def print_weights(self):
        for i, layer in enumerate(self.layers):
            print(f'Layer {i}')
            print(layer.weight)