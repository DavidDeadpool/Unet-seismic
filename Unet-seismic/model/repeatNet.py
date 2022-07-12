import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from apex import parallel
from torchsummary import summary
from SyncBN.sync_batchnorm import convert_model
from model.GC_Context_Module import ContextBlock2d

class RepeatBlock(nn.Module):
    def __init__(self, wide, use_context=False):
        super(RepeatBlock, self).__init__()

        self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(in_channels=wide, out_channels=wide, kernel_size=3, stride=1, dilation=2, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=wide * 2,out_channels=wide, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(wide)
        self.use_context = use_context
        if use_context:
            self.context = ContextBlock2d(inplanes=wide,planes=wide,pool='att',fusions=['channel_add'])


    def forward(self, x):
        pool = self.pool1(x)
        conv1 = self.conv1(pool)
        conv1 = self.relu(conv1)
        conv1 = F.interpolate(conv1,size=x.size()[2:],mode='bilinear',align_corners=True)
        cat = t.cat([x,conv1], dim=1)
        conv2 = self.conv2(cat)
        bn = self.bn(conv2)
        bn = self.relu(bn)
        bn = x + bn
        if self.use_context:
            bn = self.context(bn)
        return bn


class RepeatNet(nn.Module):
    def __init__(self, Repeat_num=24, class_num=2, wide=32, use_context=False):
        super(RepeatNet, self).__init__()
        self.wide = wide
        self.use_context = use_context
        self.repeat_num = Repeat_num
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=wide, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(wide)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = self._make_layer(Repeat_num)
        self.final_conv = nn.Conv2d(wide,class_num,kernel_size=1,stride=1,bias=False)


    def _make_layer(self,Repeat_num):
        layers=[]
        for i in range(Repeat_num):
            layers.append(RepeatBlock(wide=self.wide, use_context=self.use_context))
        layers = nn.Sequential(*layers)
        return layers

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn(x)
        x = self.relu(x)
        # x = t.cat([input,x], dim=1)
        x = self.backbone(x)
        x = self.final_conv(x)
        # x = t.sigmoid(x)
        return x

    def getNetInfo(self):
        context = ""
        if self.use_context:
            context = '_with_gc_context_module'
        return 'RepeatNet_' + str(self.repeat_num) + '_wide_' + str(self.wide) + context

if __name__ == '__main__':
    net = RepeatNet(wide=32,class_num=2).cuda()
    net = convert_model(net)
    test_data = t.zeros(4,3,256,256).cuda()
    out = net(test_data)
