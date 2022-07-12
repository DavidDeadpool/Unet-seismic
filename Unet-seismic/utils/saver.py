import torch as t
import os
import numpy as np
import tifffile as tiff
from utils.makedirs import mkdirs
from matplotlib import pyplot as plt
import cv2 

def save_image(batch_data, save_path, index, mode='train',save_mask=False,plt=plt):
    mkdirs(save_path)
    N = batch_data.shape[0]
    H, W = batch_data.shape[2], batch_data.shape[3]
    # batch_data = batch_data.reshape(N, 2, 256, 256)
    if save_mask:
        batch_data = t.sigmoid(batch_data)
        batch_data = batch_data.detach().cpu().numpy()
        batch_data = np.where(batch_data > 0.5, 1, 0)
    else:
        batch_data = batch_data.detach().cpu().numpy()
    for i in range(N):
        data = batch_data[i] * 255.0
        data = data.reshape((H, W))

        if save_mask:
            data = data.astype('uint8')
        if mode == 'train':
            plt.imsave(save_path + '/' + str(index[i]) + ".tiff", data)
        elif mode == 'test':
            plt.imsave('./predict/test/' + str(index[i]) + ".tiff", data)
        elif mode == 'predict':
            plt.imsave(save_path + '/' + str(index[i]) + ".tiff", data)
        elif mode == 'single':
            plt.imsave(save_path + '/' + str(index[i]) + ".tiff", data, cmap=plt.cm.gray)


def saver(current_metric, epoch, log, cfg, model, optimizer,just_save_stat_dict=False ,mode='train'):
    # if current_metric > best_metrics[metric_choose]:
    if True:
        log.writeLog("###############################################################\n")
        log.writeLog("Save model at {} epoch, the metric is {:.6}\n".format(epoch + 1, current_metric))
        log.writeLog("###############################################################\n")
        # best_metrics[metric_choose] = current_metric
        if mode == 'train':
            mkdirs(cfg['checkpoint_path'])
            if just_save_stat_dict:
                t.save(model.state_dict(), cfg['checkpoint_path'] + str(epoch + 1) + "_" + cfg['model_checkpoint_name'])
                # t.save(optimizer.state_dict(), cfg['checkpoint_path'] + str(epoch + 1) + "_" +cfg['optimizer_checkpoint_name'])
                t.save(optimizer.state_dict(), cfg['checkpoint_path'] + "last_" +cfg['optimizer_checkpoint_name'])
            else:
                t.save(model, cfg['checkpoint_path'] + str(epoch + 1) + "_" + cfg['model_checkpoint_name'])
                # t.save(optimizer, cfg['checkpoint_path'] + str(epoch + 1) + "_" +cfg['optimizer_checkpoint_name'])
                t.save(optimizer, cfg['checkpoint_path'] + "last_" +cfg['optimizer_checkpoint_name'])
        else:
            if just_save_stat_dict:
                t.save(model.state_dict(), './checkpoint/test/' + cfg['model_checkpoint_name'])
                t.save(optimizer.state_dict(), './checkpoint/test/' + cfg['optimizer_checkpoint_name'])
            else:
                t.save(model, './checkpoint/test/' + cfg['model_checkpoint_name'])
                t.save(optimizer, './checkpoint/test/' + cfg['optimizer_checkpoint_name'])