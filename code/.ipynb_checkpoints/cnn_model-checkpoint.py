import math
from skorch import NeuralNetClassifier
import torch
from torch import nn
import torch.nn.init as init

def initialize_weights(m): # HE initialization
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

class CNN(nn.Module):
    def __init__(self, input_size, l1_channels, l1_kernel_size, l1_padding, l1_stride, 
                 l2_channels, l2_kernel_size, l2_max_pool_kernel_size, l2_padding, l2_stride, 
                 l2_dropout, l3_dropout, l4_input, l4_dropout, l5_input, output_size, num_channels=3):
        super(CNN, self).__init__()
        self.num_channels = num_channels
        if l1_padding == 'same':
            l1_padding = l1_kernel_size // 2
        if l2_padding == 'same':
            l2_padding = l2_kernel_size //2
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, l1_channels, kernel_size=l1_kernel_size, stride=l1_stride, padding=l1_padding),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l1_channels, l2_channels, kernel_size=l2_kernel_size, stride=l2_stride, padding=l2_padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=l2_max_pool_kernel_size),
            nn.Dropout(l2_dropout)
        )

        self.l3_input = self.calculate_l3_input(input_size)
        self.fcs = nn.Sequential(
            nn.Linear(self.l3_input, l4_input),
            nn.ReLU(),
            nn.Dropout(l3_dropout),
            nn.Linear(l4_input, l5_input),
            nn.ReLU(),
            nn.Dropout(l4_dropout),
            nn.Linear(l5_input, output_size)
        )
        self.apply(initialize_weights)

    def calculate_l3_input(self, input_size):
        dummy_input = torch.zeros(1, self.num_channels, input_size, input_size)
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        return int(math.prod(
            x.shape[1:]))  # product of all the elements in the flattened vector except for the batch size of course

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fcs(x)
        return x