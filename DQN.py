import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=h, out_features=12)
        self.fc2 = nn.Linear(in_features=12, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
