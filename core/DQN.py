import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(torch.nn.Module):
    def __init__(self, input_size, outputs, neurons_number):
        super(Net, self).__init__()

        modules = [
            torch.nn.Linear(in_features=input_size, out_features=neurons_number),
            torch.nn.Linear(in_features=neurons_number, out_features=neurons_number),
            torch.nn.Linear(in_features=neurons_number, out_features=outputs),
            torch.nn.LeakyReLU(0.1)
        ]
        self.net = torch.nn.ModuleList(modules)

    def forward(self, inputs):
        for i, n in enumerate(self.net):
            inputs = n(inputs)
        return inputs


class DQN(nn.Module):

    def __init__(self, input_size, outputs, neurons_number):
        self.neurons_number = int(neurons_number)
        super(DQN, self).__init__()
        self.net = Net(input_size, outputs, neurons_number)

        if torch.cuda.is_available():
            self.net = torch.nn.DataParallel(self.net)
            print('Model:', type(self.net))
            print('Devices:', self.net.device_ids)

    def forward(self, x):
        x = self.net.forward(x)

        return x.view(x.size(0), -1)  # changing the size of the tensor
