import math
from skorch import NeuralNetClassifier
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, model_params):
        super(CNN, self).__init__()
        self.model_params = model_params
        self.conv1 = nn.Sequential(
            nn.Conv2d(model_params['num_channels'], model_params['l1_channels'],
                      kernel_size=model_params['l1_kernel_size']),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(model_params['l1_channels'], model_params['l2_channels'],
                      kernel_size=model_params['l2_kernel_size']),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=model_params['l2_max_pool_kernel_size']),
            nn.Dropout(model_params['l2_dropout'])
        )

        self.l3_input = self.calculate_l3_input(model_params['input_size'])
        self.fcs = nn.Sequential(
            nn.Linear(in_features=self.l3_input, out_features=model_params['l4_input']),
            # Adjust dimensions based on input size
            nn.ReLU(),
            nn.Dropout(model_params['l3_dropout']),
            nn.Linear(in_features=model_params['l4_input'],
                      out_features=model_params['l5_input']),
            nn.ReLU(),
            nn.Dropout(model_params['l4_dropout']),
            nn.Linear(in_features=model_params['l5_input'],
                      out_features=model_params['output_size'])
        )

    def calculate_l3_input(self, input_size):
        dummy_input = torch.zeros(1, self.model_params['num_channels'], input_size, input_size)
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