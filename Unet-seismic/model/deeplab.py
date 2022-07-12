# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 12:58:05
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-23 14:47:57
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class Bottleneck(nn.Module):
    """
    通过 _make_layer 来构造Bottleneck
    具体通道变化：
    inplanes -> planes -> expansion * planes 直连 out1
    inplanes -> expansion * planes 残差项 res
    由于多层bottleneck级连 所以inplanes = expansion * planes 
    总体结构 expansion * planes -> planes -> expansion * planes 

    注意：
    1.输出 ReLu(out1 + res)
    2.与普通bottleneck不同点在于 其中的stride是可以设置的
    3.input output shape是否相同取决于stride   
      out:[x+2rate-3]/stride + 1 
      res:[x-1]/stride + 1


    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        # inplanes -> planes -> planes * expansion

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])#64， 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])#128 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])#256 23
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        """
        block class: 未初始化的bottleneck class
        planes:输出层数
        blocks:block个数, layers
        """
        downsample = None
        # self.inplanes = 64
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        # [b, 256, 128, 128]
        x1 = self.layer1(x)

        # low_level_feat = x1

        # [b, 512, 64, 64]
        x2 = self.layer2(x1)

        # [b, 1024, 64, 64]
        x3 = self.layer3(x2)

        # [b, 2048, 64, 64]
        x4 = self.layer4(x3)

        # print("x1 {}".format(x1.shape))
        # print("x2 {}".format(x2.shape))
        # print("x3 {}".format(x3.shape))
        # print("x4 {}".format(x4.shape))

        return x1, x2, x3, x4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet_backbone(nInputChannels=3, os=16, layers=[2, 2, 3, 2], pretrained=False):
    model = ResNet(nInputChannels, Bottleneck,layers , os, pretrained=pretrained)
    return model


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor=None):
        super(UpBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample_factor = upsample_factor

    def forward(self, low_feature, high_feature):
        low_feature = self.conv1(low_feature)
        if self.upsample_factor is not None:
            high_feature = F.interpolate(high_feature, scale_factor=self.upsample_factor, mode='bilinear', align_corners=True)
        fusion = low_feature + high_feature
        fusion = self.bn1(fusion)
        fusion = self.relu(fusion)
        return fusion


class UpPath(nn.Module):
    def __init__(self):
        super(UpPath,self).__init__()
        self.block4 = UpBlock(2048, 256)
        self.block3 = UpBlock(1024, 256)
        self.block2= UpBlock(512, 256)
        self.block1 = UpBlock(256, 256, upsample_factor=2)

    def forward(self, stage1,stage2,stage3,stage4,aspp):
        stage4 = self.block4(stage4, aspp)
        stage3 = self.block3(stage3,stage4)
        stage2 = self.block2(stage2,stage3)
        stage1 = self.block1(stage1,stage2)
        stage1 = F.interpolate(stage1, scale_factor=2, mode='bilinear', align_corners=True)
        return stage1



class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=2, os=8, layers=[2,3,3,2],pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes       : {}".format(n_classes))
            print("Output stride           : {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
            print("Input shape             : {}".format("batchsize, 3, 256, 256"))
            print("Output shape            : {}".format("batchsize,{}, 256, 256".format(n_classes)))
            print("Net Layers              : {}".format(layers))

        super(DeepLabv3_plus, self).__init__()
        self.layers = layers
        # Atrous Conv
        self.resnet_features = ResNet_backbone(nInputChannels, os, layers=layers, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.uppath = UpPath()

        self.last_conv = nn.Sequential(
                                       # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       # nn.BatchNorm2d(256),
                                       # nn.ReLU(),
                                       # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       # nn.BatchNorm2d(256),
                                       # nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):#input 1, 3, 512, 512
        stage1, stage2, stage3,stage4 = self.resnet_features(input)#final_x:[1, 2048, 32, 32]  low_level_features:[1,256, 128, 128]
        x1 = self.aspp1(stage4)   #[1, 256, 64, 64]
        x2 = self.aspp2(stage4)   #[1, 256, 64, 64]
        x3 = self.aspp3(stage4)   #[1, 256, 64, 64]
        x4 = self.aspp4(stage4)   #[1, 256, 64, 64]
        x5 = self.global_avg_pool(stage4) #[1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # [256,64,64]
        x = self.uppath(stage1, stage2, stage3, stage4, x)
        # x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/2)), int(math.ceil(input.size()[-1]/2))), mode='bilinear', align_corners=True)
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # low_level_features = self.conv2(stage1)
        # low_level_features = self.bn2(low_level_features)
        # low_level_features = self.relu(low_level_features)
        #
        # x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def getNetInfo(self):
        return 'Deeplab_v3+_' + str(self.layers)



def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=2, os=8, pretrained=False, _print=True).cuda()
    model.eval()
    image = torch.randn(1, 3, 256, 256).cuda()
    for i in range(1):
        output = model(image)
        print(output.size())