import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEWithWeights(nn.Module):
    def __init__(self, weight=200):
        super(MSEWithWeights, self).__init__()
        self.weight = weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target, attention_mask):
        # output = torch.sigmoid(output)
        loss = self.mse(output, target)
        weighted = self.weight * attention_mask
        return (loss * weighted + loss).mean()


class BCEWithWeights(nn.Module):
    def __init__(self, weight=200):
        super(BCEWithWeights, self).__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target, attention_mask):
        loss = self.bce(output, target)
        weighted = self.weight * attention_mask
        return (loss * weighted + loss).mean()


class MultLoss(nn.Module):
    def __init__(self, bceweight=10, mseweight=10):
        super(MultLoss, self).__init__()
        self.mse = MSEWithWeights(mseweight)
        self.bce = BCEWithWeights(bceweight)

    def forward(self, mask, mask_target, regress, regress_target):
        attention_mask = (mask_target == 1.0).float()
        loss_mse = self.mse(regress, regress_target, attention_mask)
        loss_bce = self.bce(mask, mask_target, attention_mask)
        return loss_bce + loss_mse


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        # print(input)
        N = input.size()[0]
        smooth = 0.001
        if input.size()[1] > 1:
            # 如果input是个维度为2的矩阵，去第2维作为正例
            input = input[:, 1, :, :]
            # print(input.requires_grad)
        if len(target.size()) > 3 and target.size()[1] > 1:
            # print(target.size())
            target = target[:, 1, :, :]

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        target_flat = target_flat.float()
        intersection = input_flat * target_flat
        # print(intersection)
        # print(intersection.sum(1))
        # print(input_flat)
        # print(input_flat.sum(1))
        # print(target_flat)
        # print(target_flat.sum(1))
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):
        C = target.shape[1]
        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        dice = DiceLoss()
        totalLoss = 0
        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
        return totalLoss


class BCE_Dice(nn.Module):
    def __init__(self):
        super(BCE_Dice, self).__init__()

    def forward(self, inputs, target):
        if len(target.shape) < 4:
            target.unsqueeze_(1)
        # print(target.shape)
        re_target = torch.abs(1 - target)
        target = torch.cat((re_target, target),dim=1).float()
        dice_loss = DiceLoss()(inputs,target)
        bce_loss = nn.BCEWithLogitsLoss()(inputs,target)
        return dice_loss + bce_loss


if __name__ == '__main__':
    cri = BCE_Dice()
    inputs = torch.tensor([0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1]).cuda()

    inputs = inputs.reshape((2,2,2,2)).float()
    inputs.requires_grad = True

    target = torch.tensor([0,1,1,0,0,1,1,0]).cuda()
    target = target.reshape((2,2,2))
    # model = nn.Conv2d(2,2,kernel_size=1,bias=False).cuda()
    #
    # output = model(inputs)

    # inputs.requires_grad = True
    # target.requires_grad = True
    loss = cri(inputs, target)