import tifffile as tiff
import torch as t
import os
import numpy as np
from utils.makedirs import mkdirs
from matplotlib import pyplot as plt

def save_image(batch_data, save_path, index, mode='train'):
    N = batch_data.shape[0]
    batch_data = batch_data.detach().cpu().numpy()
    batch_data = batch_data
    for i in range(N):
        data = batch_data[i] * 255
        # data = data.astype('uint8')
        if mode == 'train':
            plt.imsave(save_path + '/' + str(index[i]) + ".jpg", data)
        elif mode == 'test':
            plt.imsave('./predict/test/' + str(index[i]) + ".jpg", data)
        elif mode == 'predict':
            mkdirs(save_path)
            plt.imsave(save_path + index[i], data)



if __name__ == '__main__':
    data = np.random.randn(8, 2, 256, 256)
    save_image(data, 'test', 0)