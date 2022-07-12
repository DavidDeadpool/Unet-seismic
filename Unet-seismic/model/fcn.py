import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class FCN_DeepLab(nn.Module):
    def __init__(self,num_class=2):
        super(FCN_DeepLab,self).__init__()

        self.backbone = tv.models.segmentation.deeplabv3_resnet50(pretrained=False)
        self.backbone.backbone.maxpool = nn.MaxPool2d(stride=2,kernel_size=2)
        final_layer = self.backbone.classifier[-1]
        final_layer_in_channel = final_layer.in_channels
        final_layer = nn.Conv2d(in_channels=final_layer_in_channel, out_channels=num_class, kernel_size=1, stride=1, bias=False)
        self.backbone.classifier[-1] = final_layer

    def forward(self, x):
        return self.backbone(x)['out']


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            for backbone_name, backbone_module in module._modules.items():
                print(backbone_name)
                x = backbone_module(x)
                if name in self.extracted_layers:
                    outputs.append(x)
        return outputs


if __name__ == '__main__':
    net = FCN_DeepLab()
    fe = FeatureExtractor(net,['layer1','layer2','layer3','layer4','classifer'])
    test_data = t.zeros(2,3,256,256)
    out = fe(test_data)
