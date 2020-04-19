import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, input_neuron, output_neuron, hidden_neurons=[]):
        super(FCNN, self).__init__()

        neurons = [input_neuron]+hidden_neurons+[output_neuron]
        model = []
        for i in range(len(neurons)-2):
            in_neuron = neurons[i]
            out_neuron = neurons[i+1]
            model.append(nn.Linear(in_neuron, out_neuron))
            model.append(nn.ReLU())
        model.append(nn.Linear(neurons[-2], neurons[-1]))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

if __name__=='__main__':
    n_batch = 32
    input_neuron = 16
    output_neuron = 2
    hidden_neurons = [8, 4]

    model = FCNN(input_neuron, output_neuron, hidden_neurons)

    for param in model.parameters():
        print(type(param), param.size())

