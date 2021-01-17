import sys

sys.path.append('..')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class AlphaZeroNet(nn.Module):
    def __init__(self, board_width, board_height, num_actions, args):
        # game params
        self.board_x, self.board_y = board_width, board_height
        self.action_size = num_actions
        self.args = args

        super(AlphaZeroNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels // 4, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(args.num_channels // 4)
        self.conv2 = nn.Conv2d(args.num_channels // 4, args.num_channels // 2, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.bn2 = nn.BatchNorm2d(args.num_channels // 2)
        self.conv3 = nn.Conv2d(args.num_channels // 2, args.num_channels, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        # self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        # self.bn4 = nn.BatchNorm2d(args.num_channels)
        self.fc1 = nn.Linear(self.args.num_channels * (self.board_x // 4) * (self.board_y // 4), 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, self.action_size)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.pool1(self.conv1(s))))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.pool2(self.conv2(s))))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        # s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        # s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))
        s = s.view(-1, self.args.num_channels * (self.board_x // 4) * (self.board_y // 4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(
            v)  # should we use tanh(2v)? the tanh is approximatly linear around 0
