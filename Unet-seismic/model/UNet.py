import torch as t
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
from model.model_pre_post import Pre_Post_Module

class block(nn.Module):
    def __init__(self,name,in_channels,out_channels,kernel_size=(3,3),padding=1):
        super(block,self).__init__()
        num_conv_in = len(in_channels)
        num_conv_out = len(out_channels)
        assert num_conv_in == num_conv_out,'the in_channels should be same with out_channels'
        for each in range(num_conv_in):
            self.add_module(name + "/conv_{}".format(each), nn.Conv2d(in_channels[each], out_channels[each],
                                                                     kernel_size=kernel_size, padding=padding,
                                                                     stride=1, bias=False))
            self.add_module(name + "/activate_{}".format(each),nn.ReLU(inplace=True))
            self.add_module(name + "/batchnorm_{}".format(each),nn.BatchNorm2d(out_channels[each]))

    def forward(self, inputs):
        x = inputs
        for each in self.children():
            x = each(x)
        return x


class compress_block(nn.Module):
    def __init__(self, name, compress_factor=1.0, in_channel=None):
        super(compress_block, self).__init__()
        self.add_module(name + "/compress_conv", nn.Conv2d(in_channel, int(in_channel * compress_factor),
                                                      kernel_size=(1, 1), stride=1, bias=False))
        self.add_module(name + "/compress_activate", nn.ReLU(inplace=True))
        self.add_module(name + "/compress_batchnormal", nn.BatchNorm2d(int(in_channel * compress_factor)))

    def forward(self, inputs):
        x = inputs
        for each in self.children():
            x = each(x)
        return x


class dwon_trans(nn.Module):
    def __init__(self,name,compress_factor=1,in_channel=None):
        super(dwon_trans,self).__init__()
        self.add_module(name + "/avg_pooling",nn.AvgPool2d(2,2))
        if in_channel is not None:
            self.add_module(name + "/dowm_compress", compress_block(name, compress_factor, in_channel))

    def forward(self, inputs):
        x = inputs
        for each in self.children():
            x = each(x)
        return x

class upsample_trans(nn.Module):
    def __init__(self, name, compress_factor=1.0, in_channel=None):
        super(upsample_trans,self).__init__()
        # self.add_module(name + "/upsample",nn.Upsample(scale_factor=2, mode='bilinear'))
        if in_channel is not None:
            self.add_module(name + "/upsample_compress", compress_block(name, compress_factor, in_channel))

    def forward(self, input):
        x = input
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        for each in self.children():
            x = each(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels=1, class_num=1):
        super(UNet, self).__init__()
        self.block1 = block("block1", [input_channels, 64, 64], [64, 64, 64])
        self.block2 = block("block2", [64, 128, 128], [128, 128, 128])
        self.block3 = block("block3", [128, 256, 256], [256, 256, 256])
        self.block4 = block("block4", [256, 512, 512], [512, 512, 512])
        self.block5 = block("block5", [512, 1024, 1024], [1024, 1024, 1024])
        self.down1 = dwon_trans("down1")
        self.down2 = dwon_trans("down2")
        self.down3 = dwon_trans("down3")
        self.down4 = dwon_trans("down4")
        self.block6 = block("block6", [1024, 512, 512], [512, 512, 512])
        self.block7 = block("block7", [512, 256, 256], [256, 256, 256])
        self.block8 = block("block8", [256, 128, 128], [128, 128, 128])
        self.block9 = block("block9", [128, 64, 64], [64, 64, 64])
        self.upsample1 = upsample_trans("upsample1", compress_factor=0.5, in_channel=1024)
        self.upsample2 = upsample_trans("upsample2", compress_factor=0.5, in_channel=512)
        self.upsample3 = upsample_trans("upsample3", compress_factor=0.5, in_channel=256)
        self.upsample4 = upsample_trans("upsample4", compress_factor=0.5, in_channel=128)
        self.classify = nn.Conv2d(in_channels=64, out_channels=class_num, kernel_size=(1,1))
        self.regress = nn.Conv2d(in_channels=64, out_channels=class_num, kernel_size=(1,1))


    def forward(self, inputs):

        conv1 = self.block1(inputs)
        conv2 = self.block2(self.down1(conv1))
        conv3 = self.block3(self.down2(conv2))
        conv4 = self.block4(self.down3(conv3))
        conv5 = self.block5(self.down4(conv4))

        conv6 = self.block6(t.cat((self.upsample1(conv5), conv4), dim=1))
        conv7 = self.block7(t.cat((self.upsample2(conv6), conv3), dim=1))
        conv8 = self.block8(t.cat((self.upsample3(conv7), conv2), dim=1))
        conv9 = self.block9(t.cat((self.upsample4(conv8), conv1), dim=1))

        out = self.classify(conv9)
        reg = self.regress(conv9)
        return out, reg

    def getNetInfo(self):
        return "UNet"


if __name__ == '__main__':
    from time import time
    x = t.zeros((4, 1, 256, 256)).cuda()
    Net = UNet().cuda()
    start = time()
    for i in range(1):
        out, reg = Net(x)
        sum = out.sum()
        sum.backward()
    end = time()
    print(end - start)
    # summary(Net, (3, 256, 256))
