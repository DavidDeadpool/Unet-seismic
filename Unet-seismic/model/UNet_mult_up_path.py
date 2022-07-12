import torch as t
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
from model.model_pre_post import Pre_Post_Module


class block(nn.Module):
    def __init__(self, name, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(block, self).__init__()
        num_conv_in = len(in_channels)
        num_conv_out = len(out_channels)
        assert num_conv_in == num_conv_out, 'the in_channels should be same with out_channels'
        for each in range(num_conv_in):
            self.add_module(name + "/conv_{}".format(each), nn.Conv2d(in_channels[each], out_channels[each],
                                                                      kernel_size=kernel_size, padding=padding,
                                                                      stride=1, bias=False))
            self.add_module(name + "/batchnorm_{}".format(each),
                            nn.BatchNorm2d(out_channels[each]))
            self.add_module(name + "/activate_{}".format(each),
                            nn.ReLU(inplace=True))

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
        self.add_module(name + "/compress_batchnormal",
                        nn.BatchNorm2d(int(in_channel * compress_factor)))
        self.add_module(name + "/compress_activate", nn.ReLU(inplace=True))

    def forward(self, inputs):
        x = inputs
        for each in self.children():
            x = each(x)
        return x


class dwon_trans(nn.Module):
    def __init__(self, name, compress_factor=1, in_channel=None):
        super(dwon_trans, self).__init__()
        self.add_module(name + "/avg_pooling", nn.AvgPool2d(2, 2))
        if in_channel is not None:
            self.add_module(name + "/dowm_compress",
                            compress_block(name, compress_factor, in_channel))

    def forward(self, inputs):
        x = inputs
        for each in self.children():
            x = each(x)
        return x


class upsample_trans(nn.Module):
    def __init__(self, name, compress_factor=1.0, in_channel=None):
        super(upsample_trans, self).__init__()
        # self.add_module(name + "/upsample",nn.Upsample(scale_factor=2, mode='bilinear'))
        if in_channel is not None:
            self.add_module(name + "/upsample_compress",
                            compress_block(name, compress_factor, in_channel))

    def forward(self, input):
        x = input
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        for each in self.children():
            x = each(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels=1, class_num=1, base_channel=32):
        super(UNet, self).__init__()
        self.base_channel = base_channel

        self.block1 = block("block1", [input_channels, base_channel], [
                            base_channel, base_channel])
        self.block2 = block(
            "block2", [base_channel, base_channel * 2], [base_channel * 2, base_channel * 2])
        self.block3 = block("block3", [
                            base_channel * 2, base_channel * 4], [base_channel * 4, base_channel * 4])
        self.block4 = block("block4", [
                            base_channel * 4, base_channel * 8], [base_channel * 8, base_channel * 8])
        self.block5 = block("block5", [
                            base_channel * 8, base_channel * 16], [base_channel * 16, base_channel * 16])
        self.down1 = dwon_trans("down1")
        self.down2 = dwon_trans("down2")
        self.down3 = dwon_trans("down3")
        self.down4 = dwon_trans("down4")

        self.block6_1 = block("block6_1", [
                              base_channel * 16, base_channel * 8], [base_channel * 8, base_channel * 8])
        self.block7_1 = block("block7_1", [
                              base_channel * 8, base_channel * 4], [base_channel * 4, base_channel * 4])
        self.block8_1 = block("block8_1", [
                              base_channel * 4, base_channel * 2], [base_channel * 2, base_channel * 2])
        self.block9_1 = block("block9_1", [base_channel, base_channel], [
                              base_channel, base_channel])
        self.block9_1_1 = block("block9_1_1", [base_channel, base_channel], [
                              base_channel, base_channel])
        
        self.block6_2 = block("block6_2", [
                              base_channel * 16, base_channel * 8], [base_channel * 8, base_channel * 8])
        self.block7_2 = block("block7_2", [
                              base_channel * 8, base_channel * 4], [base_channel * 4, base_channel * 4])
        self.block8_2 = block("block8_2", [
                              base_channel * 4, base_channel * 2], [base_channel * 2, base_channel * 2])
        self.block9_2 = block("block9_2", [base_channel, base_channel], [
                              base_channel, base_channel])
        self.block9_2_1 = block("block9_2_1", [base_channel, base_channel], [
                              base_channel, base_channel])
        
        self.upsample1_1 = upsample_trans(
            "upsample1_1", compress_factor=0.5, in_channel=base_channel * 16)
        self.upsample2_1 = upsample_trans(
            "upsample2_1", compress_factor=0.5, in_channel=base_channel * 8)
        self.upsample3_1 = upsample_trans(
            "upsample3_1", compress_factor=0.5, in_channel=base_channel * 4)
        self.upsample4_1 = upsample_trans(
            "upsample4_1", compress_factor=0.5, in_channel=base_channel * 2)
        self.upsample5_1 = upsample_trans(
            "upsample5_1", compress_factor=0.5, in_channel=base_channel * 1)
        
        self.upsample1_2 = upsample_trans(
            "upsample1_2", compress_factor=0.5, in_channel=base_channel * 16)
        self.upsample2_2 = upsample_trans(
            "upsample2_2", compress_factor=0.5, in_channel=base_channel * 8)
        self.upsample3_2 = upsample_trans(
            "upsample3_2", compress_factor=0.5, in_channel=base_channel * 4)
        self.upsample4_2 = upsample_trans(
            "upsample4_2", compress_factor=0.5, in_channel=base_channel * 2)
        self.upsample5_2 = upsample_trans(
            "upsample5_2", compress_factor=0.5, in_channel=base_channel * 1)
        
        self.classify = nn.Conv2d(
            in_channels=base_channel, out_channels=class_num, kernel_size=(1, 1))
        self.regress = nn.Conv2d(
            in_channels=base_channel, out_channels=class_num, kernel_size=(1, 1))

    def forward(self, inputs):

        conv1 = self.block1(inputs)
        conv2 = self.block2(self.down1(conv1))
        conv3 = self.block3(self.down2(conv2))
        conv4 = self.block4(self.down3(conv3))
        conv5 = self.block5(self.down4(conv4))

        conv6_1 = self.block6_1(t.cat((self.upsample1_1(conv5), conv4), dim=1))
        conv7_1 = self.block7_1(
            t.cat((self.upsample2_1(conv6_1), conv3), dim=1))
        conv8_1 = self.block8_1(
            t.cat((self.upsample3_1(conv7_1), conv2), dim=1))
        conv9_1 = self.block9_1(self.upsample4_1(conv8_1))
        conv9_1_1 = self.block9_1_1(conv9_1)

        conv6_2 = self.block6_2(t.cat((self.upsample1_2(conv5), conv4), dim=1))
        conv7_2 = self.block7_2(
            t.cat((self.upsample2_2(conv6_2), conv3), dim=1))
        conv8_2 = self.block8_2(
            t.cat((self.upsample3_2(conv7_2), conv2), dim=1))
        conv9_2 = self.block9_2(self.upsample4_2(conv8_2))
        conv9_2_1 = self.block9_2_1(conv9_2)
        #conv9_1_size = conv9_1.shape[0]
        #fc1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1).cuda()

        #OUT = t.reshape(conv9_1, (1, out_size, 256, 256))

        #out_1 = fc1(conv9_1)
        #out_2 = fc1(out_1)
        #out_3 = fc1(out_2)
        #out_final = t.reshape(out_3, (out_size, 1, 256, 256))

        #reg_size = reg.shape[0]
        #fc2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1).cuda()
        #REG = t.reshape(reg, (1, reg_size, 256, 256))
        #reg_9_1 = fc2(conv9_2)
        #reg_9_2 = fc2(reg_9_1)
        #reg_9_3 = fc2(reg_9_2)
        #reg_final = t.reshape(reg_9_3, (out_size, 1, 256, 256))

        out = self.classify(conv9_1_1)
        reg = self.regress(conv9_2_1)

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
