import torch

def onehot(target):
    re_target = abs(target - 1)
    return  torch.cat([re_target, target], dim=1)


if __name__ == '__main__':
    target = torch.ones(4,1,256,256)
    out = onehot(target)
