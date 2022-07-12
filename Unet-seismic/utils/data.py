import torch as t
import torchvision as tv
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as Trans
import numpy as np

class RemoteDate(Dataset):
    def __init__(self, dataset_path, trans=None, logFlie=None, one_hot=False, mode='train'):
        self.trans = trans
        self.dataset_images_path = dataset_path + "/images/"
        self.dataset_gt_path = dataset_path + "/gt/"
        self.dataset_images = os.listdir(self.dataset_images_path)
        self.dataset_images = [each[:-5] for each in self.dataset_images]
        self.one_hot = one_hot
        self.mode = mode
        if logFlie is not None:
            logFlie.writelines("images in dataset is {}".format(len(self.dataset_images)))


    def __getitem__(self, item):
        image_name = self.dataset_images[item]
        image = Image.open(self.dataset_images_path + image_name + ".tiff")
        gt = Image.open(self.dataset_gt_path + image_name + ".tiff")

        if self.mode == 'train':
            is_aug = np.random.random()
            if is_aug > 0.5:
                flip_method = np.random.random()
                if flip_method > 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    gt = gt.transpose(Image.FLIP_TOP_BOTTOM)

        if self.trans:
            image = self.trans(image)
        gt = np.array(gt)/255
        if self.one_hot:
            re_gt = np.abs(1 - gt)
            gt = np.stack((re_gt, gt),axis=0)
        gt = t.tensor(gt).long()
        # assert image.sum() > 1
        return image, gt


    def __len__(self):
        return len(self.dataset_images)


def getDataloader(dataset_path,batch=4,shuffle=True,io_worker=4,tran=None,get_loader=True,one_hot=False, mode='train'):
    if tran is None:
        tran = Trans.Compose([
            Trans.ToTensor()
        ])
    remotedate = RemoteDate(dataset_path, tran, one_hot=one_hot,mode=mode)
    dataloader = DataLoader(
        dataset=remotedate,
        shuffle=shuffle,
        batch_size=batch,
        num_workers=io_worker,
        pin_memory=True
    )

    if get_loader:
        return dataloader
    else:
        return remotedate

def get_train_valid_dataloader(config, get_loader=True, one_hot=False):
    trans = Trans.Compose([
        Trans.ToTensor()
    ])
    train = getDataloader(config['train_path'],config['batch'],shuffle=True,tran=trans,
                          get_loader=get_loader, one_hot=one_hot,mode='train')

    valid = getDataloader(config['valid_path'],config['batch'],shuffle=False,tran=trans,
                          get_loader=get_loader, one_hot=one_hot,mode='valid')
    return train, valid


class Predict(Dataset):
    def __init__(self, predict_path, trans=None, logFlie=None):
        self.trans = trans
        self.predict_path = predict_path
        self.files = os.listdir(predict_path)
        self.files = [each[:-5] for each in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        image_name = self.files[item]
        x, y = image_name.split("_")[0], image_name.split("_")[1]
        image = Image.open(self.predict_path + image_name + ".tiff")
        if self.trans:
            image = self.trans(image)
        return image, image_name + ".tiff"

def getPredict(predict_path, batch, io_worker=6):
    trans = Trans.Compose([
        Trans.ToTensor()
    ])
    predict_dataset = Predict(predict_path=predict_path,trans=trans)
    predict_dataloader = DataLoader(
        dataset=predict_dataset,
        shuffle=False,
        batch_size=batch,
        num_workers=io_worker,
        pin_memory=True
    )
    return predict_dataloader


if __name__ == '__main__':
    # tran = Trans.Compose([
    #     Trans.ToTensor()
    # ])
    # remotedate = RemoteDate("/media/dl/sda1/01_dataset/mass_buildings/for_train_balance/train/",trans=tran,one_hot=False)
    # (a,b) = remotedate.__getitem__(3)
    # from utils.config import getExperimentConfig
    # cfg = getExperimentConfig()
    # a,b = get_train_valid_dataloader(cfg,get_loader=False)
    pred_loader = getPredict("/media/dl/sda1/01_dataset/mass_buildings/for_train_balance/crops/", 8)


