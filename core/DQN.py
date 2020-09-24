import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):

    def __init__(self, h, w, outputs, neurons_number):
        self.neurons_number = int(neurons_number)
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=h*w, out_features=self.neurons_number)
        self.fc2 = nn.Linear(in_features=self.neurons_number, out_features=self.neurons_number)
        # self.fc3 = nn.Linear(in_features=self.neurons_number, out_features=self.neurons_number)
        self.fc4 = nn.Linear(in_features=self.neurons_number, out_features=outputs)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x.view(x.size(0), -1)  # changing the size of the tensor
