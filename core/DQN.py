import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=h*w, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        # self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=outputs)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))
        x = self.activation(self.fc4(x))
        # return F.log_softmax(x, dim=1)
        return x.view(x.size(0), -1)  # changing the size of the tensor
