import numpy as np

from Models.hebbian_learning import HebbianAbcdNN

class TimeBasedNeuromodulatedHebbianNN(HebbianAbcdNN):

    def __init__(self, input_size, output_size, hidden_sizes, env_name, lambda_decay = 0.001) -> None:
        super(TimeBasedNeuromodulatedHebbianNN, self).__init__(input_size, output_size, hidden_sizes, env_name)
        self.lammbda_decay = lambda_decay
        self.current_step = 0

    def decay_function(self):
        return np.exp(-self.lammbda_decay * self.current_step)
    
    def update_weights(self, weights):
        super().update_weights(weights)
        self.current_step = 0

    
    def apply_hebbian_rules(self, states):
        delta_weights = [layer.weight for layer in self.layers]
        modulation_coeff = self.decay_function()
        for i, layer in enumerate(self.layers):
            delta_weights[i] = self.hebbian_coeff[i][:, :, 0] * (
                self.hebbian_coeff[i][:, :, 1] * (states[i+1] @ states[i].T) +
                self.hebbian_coeff[i][:, :, 2] * states[i].repeat(1, states[i+1].shape[0]).T +
                self.hebbian_coeff[i][:, :, 3] * states[i+1].repeat(1, states[i].shape[0]) +
                self.hebbian_coeff[i][:, :, 4] 
            )
            self.layers[i].weight += modulation_coeff * delta_weights[i]
        self.current_step += 1
    
