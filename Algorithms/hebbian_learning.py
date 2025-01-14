import torch 
import torch.nn as nn
from torchsummary import summary

class StaticNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size = 10, hidden_layers = 2) -> None:
        super(StaticNN, self).__init__()

        self.hidden = []
        self.hidden.append(nn.Linear(input_size, hidden_size, bias=False))
        for i in range(1, hidden_layers):
            self.hidden.append(nn.Linear(hidden_size, hidden_size, bias=False))
        self.hidden.append(nn.Linear(hidden_size, output_size, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.ABCD = torch.Tensor(input_size, output_size, 5)

    def forward(self, x):
        for h in self.hidden[:-1]:
            x = h(x)
            x = self.relu(x)
        # y = self.hidden[-1](x)


        # correlation = self.layer.weight * self.ABCD[:,:,1]
        # pre = self.layer.weight * self.ABCD[:,:,2]
        # post = self.layer.weight * self.ABCD[:,:,3]

        # self.layer.weight += self.ABCD[:,:,0] * (correlation + pre + post + self.ABCD[:,:,4])

        return self.hidden[-1](x)
    
    def get_layer_n_weights(self, layer):
        return layer.weight.shape[0] * layer.weight.shape[1]

    
    def get_n_weights(self):
        n_weights = 0
        for h in self.hidden:
            n_weights += self.get_layer_n_weights(h)
        return n_weights

    def update_weights(self, weights):
        a = 0
        for i, h in enumerate(self.hidden):
            b = a + self.get_layer_n_weights(h)
            w = torch.reshape(weights[a:b], (h.weight.shape))
            self.hidden[i].weight = nn.Parameter(w)
            a = b

    def print_weights(self):
        for i, h in enumerate(self.hidden):
            print(f'Layer {i}')
            print(h.weight)

class HebbianAbcdNN(StaticNN):

    def __init__(self, input_size, output_size, hidden_size = 10, hidden_layers = 2) -> None:
        super(HebbianAbcdNN, self).__init__(input_size, output_size, hidden_size, hidden_layers)




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