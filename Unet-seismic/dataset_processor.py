import torch
import torchvision as tv
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as Trans
import numpy as np
import struct
import platform
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from tqdm import tqdm
from tifffile import imread

image_size = 256


class SeismicToJPEG(Dataset):
    def __init__(self, dataset_list_path, dataset_raw_path, logFlie=None, mode='train',stride=32, wave_dir_name="2500"):
        self.stride = stride
        self.dataset_vp_list = open(dataset_list_path + mode + "_vp_list.txt").readlines()
        self.dataset_seimics_wave_list = open(dataset_list_path + mode + "_seimics_wave_list.txt").readlines()
        self.dataset_raw_path = dataset_raw_path

        self.dataset_vp_list = [each[:-1] for each in self.dataset_vp_list]
        self.dataset_seimics_wave_list = [each[:-1] for each in self.dataset_seimics_wave_list]

        self.mode = mode
        if logFlie is not None:
            logFlie.writelines("images in dataset is {}".format(len(self.dataset_seimics_wave_list)))

    def __getitem__(self, item):
        vp_name = self.dataset_vp_list[item]
        seimics_wave_name = self.dataset_seimics_wave_list[item]

        vp_file = open(self.dataset_raw_path + vp_name, 'rb')
        seimics_wave_file = open(self.dataset_raw_path + seimics_wave_name, 'rb')

        vp_X = 1001
        vp_Z = 1001

        wave_x = 201
        wave_y = 5001
        # 按照字节读取vp文件
        vp_array = np.zeros((vp_X, vp_Z))
        for i in range(vp_X):
            for j in range(vp_Z):
                data = vp_file.read(4)
                elem = struct.unpack('f', data)[0]
                vp_array[j][i] = elem
        end = vp_file.read(4)
        data = []
        # 按照字节读取dat文件，并reshape成201*5001的数组
        t = seimics_wave_file.read(4)
        while end != t:
            data.append(t)
            t = seimics_wave_file.read(4)
        data = [struct.unpack('f', each)[0] for each in data]
        wave_array = np.asarray(data)
        wave_array = wave_array.reshape((wave_x, wave_y)).T
        
        vp_array = Image.fromarray(vp_array).resize((image_size, image_size))
        wave_array = Image.fromarray(wave_array).resize((image_size,image_size))
        
        vp_array = np.array(vp_array)
        wave_array = np.array(wave_array)
        
#        vp_array = (vp_array - vp_array.min())/(vp_array.max() - vp_array.min())
#        wave_array = (wave_array - wave_array.min())/(wave_array.max() - wave_array.min())

        vp_file.close()
        seimics_wave_file.close()

        vp_array = torch.tensor(vp_array).float()
        wave_array = torch.tensor(wave_array).float()
        vp_array.unsqueeze_(0)
        wave_array.unsqueeze_(0)
        vp_mask = (vp_array < 2000)

        return wave_array, vp_array, vp_mask, vp_name

    def __len__(self):
        return len(self.dataset_vp_list)

class SeismicData(Dataset):
    def __init__(self, dataset_list_path, dataset_raw_path, logFlie=None, mode='train',stride=32, wave_dir_name="2500"):
        self.stride = stride
        self.dataset_list_path = dataset_list_path
        self.dataset_vp_list = open(dataset_list_path + mode + "_vp_list.txt").readlines()
        self.dataset_seimics_wave_list = open(dataset_list_path + mode + "_seimics_wave_list.txt").readlines()
        self.dataset_raw_path = dataset_raw_path
        self.wave_dir_name = wave_dir_name
        self.dataset_vp_list = [each[:-1] for each in self.dataset_vp_list]
        self.dataset_seimics_wave_list = [each[:-1] for each in self.dataset_seimics_wave_list]

        self.mode = mode
        if logFlie is not None:
            logFlie.writelines("images in dataset is {}".format(len(self.dataset_seimics_wave_list)))

    def __getitem__(self, item):
        vp_name = self.dataset_vp_list[item].split(".")[0]

        wave_array = imread(self.dataset_list_path + "jpeg/wave/{}/{}.tiff".format(self.wave_dir_name, vp_name))
        mask_array = imread(self.dataset_list_path + "jpeg/mask/{}.tiff".format(vp_name))
        vp_array = imread(self.dataset_list_path + "jpeg/vp/{}.tiff".format(vp_name))

        assert (wave_array[:, :, 0] == wave_array[:, :, 1]).sum() == image_size * image_size
        assert (mask_array[:, :, 0] == mask_array[:, :, 1]).sum() == image_size * image_size
        assert (vp_array[:, :, 0] == vp_array[:, :, 1]).sum() == image_size * image_size

        wave_array = wave_array[:, :, 0] / 255.0
        mask_array = mask_array[:, :, 0] / 255.0
        vp_array = vp_array[:, :, 0] / 255.0

        # wave_array = wave_array.reshape((1, image_size, image_size))
        # mask_array = mask_array.reshape((1, image_size, image_size))
        # vp_array = vp_array.reshape((1, image_size, image_size))

        wave_array = torch.tensor(wave_array).float()
        mask_array = torch.tensor(mask_array).float()
        vp_array = torch.tensor(vp_array).float()
        wave_array.unsqueeze_(0)
        mask_array.unsqueeze_(0)
        vp_array.unsqueeze_(0)

        return wave_array, mask_array, vp_array, vp_name

    def __len__(self):
        return len(self.dataset_vp_list)


class SeismicMultSourceData(Dataset):
    def __init__(self, dataset_list_path, dataset_raw_path, logFlie=None, mode='train',stride=32, wave_dir_name="2500"):
        self.stride = stride
        self.dataset_list_path = dataset_list_path
        self.dataset_vp_list = open(dataset_list_path + mode + "_vp_list.txt").readlines()
        self.dataset_seimics_wave_list = open(dataset_list_path + mode + "_seimics_wave_list.txt").readlines()
        self.dataset_raw_path = dataset_raw_path

        self.dataset_vp_list = [each[:-1] for each in self.dataset_vp_list]
        self.dataset_seimics_wave_list = [each[:-1] for each in self.dataset_seimics_wave_list]

        self.mode = mode
        if logFlie is not None:
            logFlie.writelines("images in dataset is {}".format(len(self.dataset_seimics_wave_list)))

    def __getitem__(self, item):
        vp_name = self.dataset_vp_list[item].split(".")[0]

        wave_array_500 = imread(self.dataset_list_path + "jpeg/wave/500/{}.tiff".format(vp_name))
        wave_array_1500 = imread(self.dataset_list_path + "jpeg/wave/1500/{}.tiff".format(vp_name))
        wave_array_2500 = imread(self.dataset_list_path + "jpeg/wave/2500/{}.tiff".format(vp_name))
        wave_array_3500 = imread(self.dataset_list_path + "jpeg/wave/3500/{}.tiff".format(vp_name))
        wave_array_4500 = imread(self.dataset_list_path + "jpeg/wave/4500/{}.tiff".format(vp_name))

        mask_array = imread(self.dataset_list_path + "jpeg/mask/{}.tiff".format(vp_name))
        vp_array = imread(self.dataset_list_path + "jpeg/vp/{}.tiff".format(vp_name))

        assert (wave_array_500[:, :, 0] == wave_array_500[:, :, 1]).sum() == image_size * image_size
        assert (wave_array_1500[:, :, 0] == wave_array_1500[:, :, 1]).sum() == image_size * image_size
        assert (wave_array_2500[:, :, 0] == wave_array_2500[:, :, 1]).sum() == image_size * image_size
        assert (wave_array_3500[:, :, 0] == wave_array_3500[:, :, 1]).sum() == image_size * image_size
        assert (wave_array_4500[:, :, 0] == wave_array_4500[:, :, 1]).sum() == image_size * image_size
        assert (mask_array[:, :, 0] == mask_array[:, :, 1]).sum() == image_size * image_size
        assert (vp_array[:, :, 0] == vp_array[:, :, 1]).sum() == image_size * image_size

        wave_array_500  = wave_array_500 [:, :, 0] / 255.0
        wave_array_1500 = wave_array_1500[:, :, 0] / 255.0
        wave_array_2500 = wave_array_2500[:, :, 0] / 255.0
        wave_array_3500 = wave_array_3500[:, :, 0] / 255.0
        wave_array_4500 = wave_array_4500[:, :, 0] / 255.0

        mask_array = mask_array[:, :, 0] / 255.0
        vp_array = vp_array[:, :, 0] / 255.0

        wave_array_500  = wave_array_500.reshape((1, image_size, image_size))
        wave_array_1500 = wave_array_1500.reshape((1, image_size, image_size))
        wave_array_2500 = wave_array_2500.reshape((1, image_size, image_size))
        wave_array_3500 = wave_array_3500.reshape((1, image_size, image_size))
        wave_array_4500 = wave_array_4500.reshape((1, image_size, image_size))

        # mask_array = mask_array.reshape((1, image_size, image_size))
        # vp_array = vp_array.reshape((1, image_size, image_size))

        wave_array_500  = torch.tensor(wave_array_500).float()
        wave_array_1500 = torch.tensor(wave_array_1500).float()
        wave_array_2500 = torch.tensor(wave_array_2500).float()
        wave_array_3500 = torch.tensor(wave_array_3500).float()
        wave_array_4500 = torch.tensor(wave_array_4500).float()

        wave_array = torch.cat([wave_array_500, wave_array_1500, wave_array_2500, wave_array_3500, wave_array_4500], dim=0)
        mask_array = torch.tensor(mask_array).float()
        vp_array = torch.tensor(vp_array).float()
        mask_array.unsqueeze_(0)
        vp_array.unsqueeze_(0)

        return wave_array, mask_array, vp_array, vp_name

    def __len__(self):
        return len(self.dataset_vp_list)



def getDataloader(dataset_path, dataset_raw_path, SeismicData=SeismicData,batch=4,shuffle=True,io_worker=0,get_loader=True, mode='train', wave_dir_name="2500"):
    if platform.system() == 'Linux':
        io_worker = cpu_count()//2
    dataset = SeismicData(dataset_path, dataset_raw_path=dataset_raw_path,mode=mode, wave_dir_name="2500")
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch,
        num_workers=io_worker,
        pin_memory=True
    )
    if get_loader:
        return dataloader
    else:
        return dataset


def get_train_valid_dataloader(config, get_loader=True, SeismicData=SeismicData, wave_dir_name="2500"):
    train = getDataloader(dataset_path=config['train_path'],dataset_raw_path=config['dataset_raw_path'],batch=config['batch'],shuffle=True,
                          get_loader=get_loader,mode='train', SeismicData=SeismicData, wave_dir_name="2500")
    valid = getDataloader(dataset_path=config['valid_path'],dataset_raw_path=config['dataset_raw_path'],batch=config['batch'],shuffle=False,
                          get_loader=get_loader,mode='test', SeismicData=SeismicData, wave_dir_name="2500")
    return train, valid


def generate_jpeg():
    try:
        os.makedirs("jpeg/vp/")
    except FileExistsError:
        print("目录vp已存在")
    try:
        os.makedirs("jpeg/mask/")
    except FileExistsError:
        print("目录mask已存在")
    try:
        os.makedirs("jpeg/wave/")
    except FileExistsError:
        print("目录wave已存在")

    config = {'train_path': './',
              'valid_path': './',
              'dataset_raw_path': './raw_data/',
              'batch': 1}
#    dataset = SeismicData("./", dataset_raw_path='./raw_data/', mode='train')

    a, b = get_train_valid_dataloader(config, True, SeismicData=SeismicToJPEG)

    for i, (wave_array, vp_array, vp_mask, vp_name) in enumerate(tqdm(a)):
        wave_array = wave_array.numpy()
        vp_array = vp_array.numpy()
        vp_mask = vp_mask.numpy()

        wave_array = wave_array.reshape((image_size, image_size))
        vp_array = vp_array.reshape((image_size, image_size))
        vp_mask = vp_mask.reshape((image_size, image_size))
        vp_name = vp_name[0].split(".")[0]

        plt.imsave("jpeg/wave/{}.tiff".format(vp_name), wave_array, vmin=-1000, vmax=1000, cmap=plt.cm.gray)
        plt.imsave("jpeg/mask/{}.tiff".format(vp_name), vp_mask, cmap=plt.cm.gray)
        plt.imsave("jpeg/vp/{}.tiff".format(vp_name), vp_array, cmap=plt.cm.gray)

    for i, (wave_array, vp_array, vp_mask, vp_name) in enumerate(tqdm(b)):
        wave_array = wave_array.numpy()
        vp_array = vp_array.numpy()
        vp_mask = vp_mask.numpy()

        wave_array = wave_array.reshape((image_size, image_size))
        vp_array = vp_array.reshape((image_size, image_size))
        vp_mask = vp_mask.reshape((image_size, image_size))
        vp_name = vp_name[0].split(".")[0]

        plt.imsave("jpeg/wave/{}.tiff".format(vp_name), wave_array, vmin=-1000, vmax=1000, cmap=plt.cm.gray)
        plt.imsave("jpeg/mask/{}.tiff".format(vp_name), vp_mask, cmap=plt.cm.gray)
        plt.imsave("jpeg/vp/{}.tiff".format(vp_name), vp_array, cmap=plt.cm.gray)


if __name__ == '__main__':
    # generate_jpeg()
    config = {'train_path': './mult_source/',
              'valid_path': './',
              'dataset_raw_path': './raw_data/',
              'batch': 1}
    # dataset = SeismicData("./", dataset_raw_path='./raw_data/', mode='train')
    a, b = get_train_valid_dataloader(config, True, SeismicData=SeismicMultSourceData)
    
    c,d,e,f = a.dataset.__getitem__(0)
    c.squeeze_()
    d.squeeze_()
    e.squeeze_()
    c = c.numpy()
    d = d.numpy()
    e = e.numpy()
    plt.imsave("test.tif",c,vmin=-1000,vmax=1000,cmap=plt.cm.gray)
    
