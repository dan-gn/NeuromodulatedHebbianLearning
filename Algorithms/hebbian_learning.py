import numpy as np
import torch 
import torch.nn as nn
from torchsummary import summary

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

class HebbianAbcdNN(StaticNN):

    def __init__(self, input_size, output_size, hidden_sizes, env_name) -> None:
        super(HebbianAbcdNN, self).__init__(input_size, output_size, hidden_sizes)
        self.hebbian_coeff = [torch.ones((layer.weight.shape[0], layer.weight.shape[1], 5)) for layer in self.layers]
        self.env_name = env_name

    def update_hebbian(self, params):
        a = 0
        for i, layer in enumerate(self.layers):
            b = a + self.get_layer_n_weights(layer) * 5
            self.hebbian_coeff[i] = torch.reshape(params[a:b], ((layer.weight.shape[0], layer.weight.shape[1], 5)))
            a = b

    def get_weights(self):
        weights = torch.ones((self.get_n_weights()))
        a = 0
        for i, layer in enumerate(self.layers):
            b = a + self.get_layer_n_weights(layer)
            weights[a:b] = torch.flatten(layer.weight)
            a = b
        return weights

    def apply_hebbian_rules(self, states):        
        delta_weights = [layer.weight for layer in self.layers]
        # print('Layers')
        # for i, layer in enumerate(self.layers):
        #     print(f'Layer {i}, {layer.weight.shape}')
        # print('States')
        # for i, state in enumerate(states):
        #     print(f'State {i}, {state.shape}')
        # print('Coeff')
        # for i, coeff in enumerate(self.hebbian_coeff):
        #     print(f'Coeff {i}, {coeff.shape}')
        for i, layer in enumerate(self.layers):
            # print('layer', i, layer.weight.shape)
            for j, row in enumerate(layer.weight):
                # print('row', j, row.shape)
                for k, column in enumerate(row):
                    # print('column', k, column.shape)
                    delta_weights[i][j, k] = self.hebbian_coeff[i][j, k, 0] * (
                        self.hebbian_coeff[i][j, k, 1] * states[i][k] * states[i+1][j] + 
                        self.hebbian_coeff[i][j, k, 2] * states[i][k] +
                        self.hebbian_coeff[i][j, k, 3] * states[i+1][j] +
                        self.hebbian_coeff[i][j, k, 4])
            self.layers[i].weight += delta_weights[i]
        



    def forward(self, x):
        states = [x]
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            states.append(x)
        y = self.layers[-1](x)

        if self.env_name == 'CartPole-v1':
            states.append(torch.sigmoid(y))
        elif self.env_name == 'MountainCar-v0':
            states.append(nn.functional.hardtanh(y, 0, 2))
        elif self.env_name == 'LunarLander-v3':
            states.append(nn.functional.hardtanh(y, 0, 3))
        self.apply_hebbian_rules(states)

        return y



if __name__ == "__main__":
    model = StaticNN(input_size=4, output_size=1)
    model(torch.ones((4)))
    print(model.get_n_weights())
    w = torch.arange(50, dtype=torch.float32)
    print(w)
    model.update_weights(w)
    model.print_weights()


    # print(summary(model, (4, 1), device='cpu'))
    # # Calculate and print number of parameters per layer
    # for name, param in model.named_parameters():
    #     if param.requires_grad:  # Only include trainable parameters
    #         num_params = param.numel()
    #         print(f"Layer: {name}, Number of parameters: {num_params}")