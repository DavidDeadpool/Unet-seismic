# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torchsummary import summary
from model.GC_Context_Module import ContextBlock2d
from model.deeplab import ASPP_module
from model.model_pre_post import Pre_Post_Module


class ASPP(nn.Module):
    def __init__(self, inplane, outplane):
        super(ASPP,self).__init__()
        self.aspp1 = ASPP_module(inplane, outplane, 1)
        self.aspp2 = ASPP_module(inplane, outplane, 3)
        self.aspp3 = ASPP_module(inplane, outplane, 5)
        self.aspp4 = ASPP_module(inplane, outplane, 7)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplane, outplane, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(outplane),
                                             nn.ReLU())
        self.reduce = nn.Conv2d(inplane * 5,inplane,kernel_size=1,stride=1,bias=False)

    def forward(self, inputs):
        x1 = self.aspp1(inputs)
        x2 = self.aspp2(inputs)
        x3 = self.aspp3(inputs)
        x4 = self.aspp4(inputs)
        x5 = self.global_avg_pool(inputs) #[1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.reduce(x)
        return x


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

        self._init_weight()

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features,mode='pool'):
        super(_Transition, self).__init__()
        self.mode = mode
        self.main = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features,kernel_size=1, stride=1, bias=False)
        )
        if mode=='pool':
            self.resize = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
            self.reduce = nn.Conv2d(num_output_features, num_output_features//2 ,kernel_size=1,stride=1,bias=False)
        else:
            self.resize = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(num_output_features, num_output_features//2 ,kernel_size=1,stride=1,bias=False)
            )
        self._init_weight()

    def forward(self, inputs):
        inputs = self.main(inputs)
        resize = self.resize(inputs)
        if self.mode == 'pool':
            reduce = self.reduce(inputs)
            return resize, reduce
        else:
            return resize

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False, use_context=False, fusion=['channel_mul']):
        super(_DenseBlock, self).__init__()
        self.use_context = use_context
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        if use_context:
            self.gc_context =  ContextBlock2d(num_layers * growth_rate + num_input_features,num_layers * growth_rate + num_input_features,'att',fusions=fusion)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if 'denselayer' in name:
                new_features = layer(*features)
                features.append(new_features)
        if self.use_context:
            out = torch.cat(features, 1)
            context = self.gc_context(out)
            return context
        return torch.cat(features, 1)


class DenseUNet(nn.Module):
    def __init__(self,num_block_layer, block_out_channls, growth_rate, efficient, class_num=1, drop_out=0, use_context=False, use_aspp=False, gc_fusion_method=['channel_mul']):
        super(DenseUNet,self).__init__()
        self.num_block_layer = num_block_layer
        self.growth_rate = growth_rate
        self.block_out_channls = block_out_channls
        self.use_context = use_context
        self.gc_fusion_method = gc_fusion_method
        # self.pre = Pre_Post_Module(input_size=[5001, 201], output_size=[256, 256])
        self.add_module("conv1",nn.Conv2d(in_channels=1,out_channels=block_out_channls,kernel_size=5,padding=2))

        self.add_module("dense_block1",_DenseBlock(num_layers=num_block_layer,num_input_features=block_out_channls,
                                                   bn_size=2,growth_rate=growth_rate,fusion=gc_fusion_method,
                                                   efficient=efficient,drop_rate=drop_out,use_context=use_context))
        self.add_module("dense_block2",_DenseBlock(num_layers=num_block_layer,num_input_features=block_out_channls,
                                                   bn_size=2,growth_rate=growth_rate,fusion=gc_fusion_method,
                                                   efficient=efficient,drop_rate=drop_out,use_context=use_context))
        self.add_module("dense_block3",_DenseBlock(num_layers=num_block_layer,num_input_features=block_out_channls,
                                                   bn_size=2,growth_rate=growth_rate,fusion=gc_fusion_method,
                                                   efficient=efficient,drop_rate=drop_out,use_context=use_context))
        self.add_module("dense_block4",_DenseBlock(num_layers=num_block_layer,num_input_features=block_out_channls,
                                                   bn_size=2,growth_rate=growth_rate,fusion=gc_fusion_method,
                                                   efficient=efficient,drop_rate=drop_out,use_context=use_context))
        self.add_module("dense_block5",_DenseBlock(num_layers=num_block_layer,num_input_features=block_out_channls,
                                                   bn_size=2,growth_rate=growth_rate,fusion=gc_fusion_method,
                                                   efficient=efficient,drop_rate=drop_out,use_context=use_context))
        self.add_module("pooling_1", _Transition(num_block_layer * growth_rate + block_out_channls, block_out_channls))
        self.add_module("pooling_2", _Transition(num_block_layer * growth_rate + block_out_channls, block_out_channls))
        self.add_module("pooling_3", _Transition(num_block_layer * growth_rate + block_out_channls, block_out_channls))
        self.add_module("pooling_4", _Transition(num_block_layer * growth_rate + block_out_channls, block_out_channls))

        self.add_module("up_1", _Transition(num_block_layer * growth_rate + block_out_channls, block_out_channls, mode='up'))
        self.add_module("up_2", _Transition(num_block_layer * growth_rate + block_out_channls, block_out_channls, mode='up'))
        self.add_module("up_3", _Transition(num_block_layer * growth_rate + block_out_channls, block_out_channls, mode='up'))
        self.add_module("up_4", _Transition(num_block_layer * growth_rate + block_out_channls, block_out_channls, mode='up'))

        self.add_module("dense_block6",_DenseBlock(num_layers=num_block_layer,num_input_features=block_out_channls,
                                                   bn_size=2,growth_rate=growth_rate,fusion=gc_fusion_method,
                                                   efficient=efficient,drop_rate=drop_out,use_context=use_context))
        self.add_module("dense_block7",_DenseBlock(num_layers=num_block_layer,num_input_features=block_out_channls,
                                                   bn_size=2,growth_rate=growth_rate,fusion=gc_fusion_method,
                                                   efficient=efficient,drop_rate=drop_out,use_context=use_context))
        self.add_module("dense_block8",_DenseBlock(num_layers=num_block_layer,num_input_features=block_out_channls,
                                                   bn_size=2,growth_rate=growth_rate,fusion=gc_fusion_method,
                                                   efficient=efficient,drop_rate=drop_out,use_context=use_context))
        self.add_module("dense_block9",_DenseBlock(num_layers=num_block_layer,num_input_features=block_out_channls,
                                                   bn_size=2,growth_rate=growth_rate,fusion=gc_fusion_method,
                                                   efficient=efficient,drop_rate=drop_out,use_context=use_context))

        self.add_module("classifier",nn.Conv2d(num_block_layer * growth_rate + block_out_channls, class_num, kernel_size=1, stride=1))
        self.add_module("fusion_1", nn.Conv2d(num_block_layer * growth_rate + block_out_channls, block_out_channls, kernel_size=1, stride=1))
        self.add_module("fusion_2", nn.Conv2d(num_block_layer * growth_rate + block_out_channls, block_out_channls, kernel_size=1, stride=1))
        self.add_module("fusion_3", nn.Conv2d(num_block_layer * growth_rate + block_out_channls, block_out_channls, kernel_size=1, stride=1))
        self.add_module("fusion_4", nn.Conv2d(num_block_layer * growth_rate + block_out_channls, block_out_channls, kernel_size=1, stride=1))
        self.use_aspp = use_aspp
        # self.post = Pre_Post_Module(input_size=[256, 256], output_size=[1001, 1001])
        if use_aspp:
            self.add_module("ASPP", ASPP(num_block_layer * growth_rate + block_out_channls, num_block_layer * growth_rate + block_out_channls))
        self._init_weight()

    def forward(self, x):
        # x = self.pre(x)
        x = self.conv1(x)
        block1 = self.dense_block1(x)
        pool1, fusion1 = self.pooling_1(block1)
        block2 = self.dense_block2(pool1)
        pool2, fusion2 = self.pooling_2(block2)
        block3 = self.dense_block3(pool2)
        pool3, fusion3 = self.pooling_3(block3)
        block4 = self.dense_block4(pool3)
        pool4, fusion4 = self.pooling_4(block4)

        block5 = self.dense_block5(pool4)
        if self.use_aspp:
            block5 = self.ASPP(block5)
        up1 = self.up_1(block5)

        up1 = torch.cat([up1, fusion4], dim=1)

        block6 = self.dense_block6(up1)
        up2 = self.up_2(block6)
        up2 = torch.cat([up2, fusion3], dim=1)

        block7 = self.dense_block7(up2)
        up3 = self.up_3(block7)
        up3 = torch.cat([up3, fusion2], dim=1)

        block8 = self.dense_block8(up3)
        up4 = self.up_4(block8)
        up4 = torch.cat([up4, fusion1], dim=1)

        block9 = self.dense_block9(up4)
        out = self.classifier(block9)
        # out = self.post(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def getNetInfo(self):
        text = [self.num_block_layer, self.growth_rate, self.block_out_channls]
        text = [str(x) for x in text]
        if self.use_context:
            text.append("with_gc")
            text.append(str(self.gc_fusion_method))
        if self.use_aspp:
            text.append("with_aspp")
        text = '_'.join(text)
        return "DenseNet_" + text


if __name__ == '__main__':
    from time import time
    start = time()
    x = torch.zeros((4, 1, 256, 256)).cuda()
    Net = DenseUNet(class_num=2, block_out_channls=64, num_block_layer=4, efficient=False, use_context=False,growth_rate=12, use_aspp=False,gc_fusion_method=['channel_mul','channel_add']).cuda()
    out = Net(x)
    # for i in range(1):
    #
    #     sum = out.sum()
    #     sum.backward()
    # end = time()
    # print(end - start)
    # summary(Net,(3,128,128,))