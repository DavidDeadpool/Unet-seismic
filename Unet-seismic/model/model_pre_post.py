import torch as t
import torch.nn as nn
from torch.nn import functional as F


class Pre_Post_Module(nn.Module):
    def __init__(self, input_size, output_size):
        super(Pre_Post_Module, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=output_size[0],
                               kernel_size=(input_size[0], 1), padding=0, bias=True, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=output_size[1],
                               kernel_size=(1, input_size[1]), padding=0, bias=True, stride=1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu(x)
        x = x.permute(0,2,1,3)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu(x)
        x = x.permute(0,3,1,2)
        # print(x.shape)
        return x

if __name__ == '__main__':
    i = t.zeros((1,1,256,256))
    net = Pre_Post_Module(input_size=[256,256], output_size=[1001,1001])
    o = net(i)


