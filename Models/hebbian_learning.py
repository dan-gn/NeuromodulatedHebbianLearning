import numpy as np
import torch 
import torch.nn as nn
from Models.static_neural_network import StaticNN

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
            # for j, row in enumerate(layer.weight):
            #     # print('row', j, row.shape)
            #     for k, column in enumerate(row):
            #         # print('column', k, column.shape)
            #         delta_weights[i][j, k] = self.hebbian_coeff[i][j, k, 0] * (
            #             self.hebbian_coeff[i][j, k, 1] * states[i][k] * states[i+1][j] + 
            #             self.hebbian_coeff[i][j, k, 2] * states[i][k] +
            #             self.hebbian_coeff[i][j, k, 3] * states[i+1][j] +
            #             self.hebbian_coeff[i][j, k, 4])
                    
            delta_weights[i] = self.hebbian_coeff[i][:, :, 0] * (
                self.hebbian_coeff[i][:, :, 1] * (states[i+1] @ states[i].T) +
                self.hebbian_coeff[i][:, :, 2] * states[i].repeat(1, states[i+1].shape[0]).T +
                self.hebbian_coeff[i][:, :, 3] * states[i+1].repeat(1, states[i].shape[0]) +
                self.hebbian_coeff[i][:, :, 4] 
            )
            self.layers[i].weight += delta_weights[i]

    def forward(self, x):
        states = [x.unsqueeze(1)]
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            states.append(x.unsqueeze(1))
        y = self.layers[-1](x)

        if self.env_name == 'CartPole-v1':
            states.append(torch.sigmoid(y).unsqueeze(1))
        elif self.env_name == 'MountainCar-v0':
            states.append(nn.functional.hardtanh(y, 0, 2).unsqueeze(1))
        elif self.env_name == 'LunarLander-v3':
            states.append(nn.functional.hardtanh(y, 0, 3).unsqueeze(1))
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
