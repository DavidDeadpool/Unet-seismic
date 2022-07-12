import torch as t
import torch.nn as nn
import argparse
import time
import numpy as np
import tifffile as tiff
import os
from utils.makedirs import mkdirs
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.loss import DiceLoss
from utils.config import setup_seed
from utils.config import getExperimentConfig
from utils.reg_metrics import metric_function as reg_mf
from utils.reg_metrics import record_epoch_metric as reg_rm
from utils.reg_metrics import get_Init_Metrics as reg_gim
from utils.seg_metrics import get_Init_Metrics as seg_gim
from utils.seg_metrics import metric_function as seg_mf
from utils.seg_metrics import record_epoch_metric as seg_rm
from utils.mult_metrics import get_Init_Metrics as mult_gim
from utils.mult_metrics import metric_function as mult_mf
from utils.mult_metrics import record_epoch_metric as mult_rm
from SyncBN.sync_batchnorm import convert_model
from utils.backup_train import backup_train
from utils.saver import saver, save_image
from utils.draw_pic import drawLogFile

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

setup_seed(2019)  # 设置随机数种子用于结果复现

parser = argparse.ArgumentParser(description='PyTorch Seg Example')
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--name', default='debug', type=str)
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--base_channel', default=32, type=int)
parser.add_argument('--image_channel', default=1, type=int)
parser.add_argument('--dataset', type=str)
parser.add_argument('--pretrain_model', default='', type=str)
parser.add_argument('--pretrain_optim', default='', type=str)
args = parser.parse_args()

mode = args.mode  # 控制程序的运行方式，只有当mode为train时，程序才会正常保存log和模型
name = args.name
cmd_epoch = args.epoch
pre_model = args.pretrain_model
pre_optim = args.pretrain_optim

cfg = getExperimentConfig(experiment_name=name, mode=mode, dataset=args.dataset, batch=args.batch, image_channel=args.image_channel, base_channel=args.base_channel)

# 备份代码到log_path
backup_train('./utils/config.py', cfg['log_path'] + 'config_bak.py')
backup_train('train.py', cfg['log_path'] + 'train_bak.py')

if cmd_epoch > 0:
    cfg['epoch'] = cmd_epoch
if pre_model != '':
    cfg['pre_trained_model'] = pre_model
if pre_optim != '':
    cfg['pre_trained_optim'] = pre_optim

model = cfg['model']
model = convert_model(model)
optimizer = cfg['optimizer']

train_dataloader = cfg['train_dataloader']
valid_dataloader = cfg['valid_dataloader']

log = cfg['logger']

if cfg['task_class'] == 'reg':
    best_metrics = reg_gim()
    metric_function = reg_mf
    record_epoch_metric = reg_rm
    metric_choose = 'mae'
    loss_function = cfg['loss_function_reg']
elif cfg['task_class'] == 'seg':
    best_metrics = seg_gim()
    metric_function = seg_mf
    record_epoch_metric = seg_rm
    metric_choose = 'f1'
    loss_function = cfg['loss_function_seg']
elif cfg['task_class'] == 'mult':
    best_metrics = mult_gim()
    metric_function = mult_mf
    record_epoch_metric = mult_rm
    metric_choose = 'f1'
    loss_function = cfg['loss_function_mult']
else:
    print("task class error,should be 'reg' or 'seg' or 'mult!'")
    raise ValueError

log.writeLog("model       : {}\n".format(model.__class__))
log.writeLog("optimizer   : {}\n".format(optimizer.__class__))
log.writeLog("loss fun    : {}\n".format(loss_function.__class__))
log.writeLog("batch       : {}\n".format(cfg['batch']))
log.writeLog("epoch       : {}\n".format(cfg['epoch']))
log.writeLog("base channel: {}\n".format(args.base_channel))
log.writeLog("dataset path: {}\n".format(cfg['train_path']))
log.writeLog("metric      : {}\n".format(metric_choose))

if t.cuda.is_available():
    model = model.cuda()
    if t.cuda.device_count() > 1:
        log.writeLog("Use multi GPU, GPU number is {}\n".format(t.cuda.device_count()))
        model = nn.DataParallel(model)

if cfg['load_pretrain_model'] or pre_model != '':
    log.writeLog("Load pretrain model {} \n".format(cfg['pre_trained_model']))
    try:
        model.load_state_dict(t.load(cfg['pre_trained_model']))
        log.writeLog("Load Pretrain model static {} success".format(cfg['pre_trained_model']))
    except:
        model = t.load(cfg['pre_trained_model'])
        log.writeLog("Load Pretrain model {} success".format(cfg['pre_trained_model']))

if cfg['load_pretrain_optim'] or pre_optim != '':
    log.writeLog("Load pretrain optimizer {} \n".format(cfg['pre_trained_optim']))
    try:
        optimizer.load_state_dict(t.load(cfg['pre_trained_optim']))
        log.writeLog("Load Pretrain optim static {} success".format(cfg['pre_trained_optim']))
    except:
        optimizer = t.load(cfg['pre_trained_optim'])
        log.writeLog("Load Pretrain optim  {} success".format(cfg['pre_trained_optim']))


def train(epoch):
    log.writeLog("\nstart train\n")
    log.writeLog("output log to {} \n".format(cfg['log_path']))
    iter_count = 1
    for each in range(epoch):
        print("epoch {}\n".format(each))
        model.train()
        start_time = time.time()
        epoch_metrics = {}
        if cfg['task_class'] == 'reg':
            epoch_metrics = reg_gim()
        elif cfg['task_class'] == 'seg':
            epoch_metrics = seg_gim()
        elif cfg['task_class'] == 'mult':
            epoch_metrics = mult_gim()
        train_loss = 0.0
        train_metric = 0.0
        for i, (data, mask_target, regress_target, namelist) in enumerate(tqdm(train_dataloader, ascii=True)):
            if isinstance(loss_function, nn.BCEWithLogitsLoss):
                mask_target = mask_target.float()
            if t.cuda.is_available():
                data = data.cuda()
                regress_target = regress_target.cuda()
                mask_target = mask_target.cuda()
            mask, regress = model(data)

            if i % 1 == 0:
                save_image(mask, cfg['predict_path'] + "train/mask/", namelist, mode=mode, save_mask=True, plt=tiff)
                save_image(regress, cfg['predict_path'] + "train/regress/", namelist, mode=mode)

                save_image(regress, cfg['predict_path'] + "train/single/", namelist, mode='single')
            optimizer.zero_grad()
            loss = loss_function(mask, mask_target, regress, regress_target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()
            metrics_value = metric_function(mask, mask_target, regress, regress_target)
            epoch_metrics = record_epoch_metric(epoch_metrics, metrics_value)
            train_metric = train_metric + metrics_value[metric_choose]
            log.add_scalar("train/train_iter_loss", loss.item(), iter_count)
            iter_count += 1

        end_time = time.time()
        spend_time = end_time - start_time
        # train_loss = train_loss / (i + 1)
        # train_metric = train_metric / (i + 1)
        epoch_metrics['loss'] = train_loss
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= (i + 1)
            log.addRecord('train_' + k, epoch_metrics[k])
            log.drawRecord()
        log.writeLog("_______________________________________________________________\n")
        log.writeLog("- Epoch {} time    - {}  s\n".format(each + 1, spend_time))
        log.writeLog("  train loss       - {:.6}\n".format(train_loss))
        log.writeLog("  train metric     - {:.6}\n".format(train_metric))
        log.writeLog("  train all metric - {}\n".format(epoch_metrics))
        valid_loss, valid_metric = valid(valid_dataloader, each)

        log.add_scalars("train/train_val_loss", {"train_loss": train_loss,
                                                    "valid_loss": valid_loss}, each)
        log.add_scalars("train/train_val_metrics", {"train_metric": train_metric,
                                                       "valid_metric": valid_metric}, each)
        log.add_scalars('train/all_train_metrics', epoch_metrics, each)
        saver(model=model,optimizer=optimizer,cfg=cfg,
              log=log,epoch=each, current_metric=valid_metric,mode=mode,just_save_stat_dict=True)


def valid(valid_dataloader, epoch, dotaloader_des='valid'):
    # model.eval()
    for each in range(1):
        valid_loss = 0.0
        valid_metric = 0.0
        epoch_metrics = {}
        if cfg['task_class'] == 'reg':
            epoch_metrics = reg_gim()
        elif cfg['task_class'] == 'seg':
            epoch_metrics = seg_gim()
        elif cfg['task_class'] == 'mult':
            epoch_metrics = mult_gim()
        with t.no_grad():
            for i, (data, mask_target, regress_target, namelist) in enumerate(tqdm(valid_dataloader)):
                if isinstance(loss_function, nn.BCEWithLogitsLoss):
                    mask_target = mask_target.float()
                if t.cuda.is_available():
                    data = data.cuda()
                    regress_target = regress_target.cuda()
                    mask_target = mask_target.cuda()
                mask, regress = model(data)
                if i % 1 == 0:
                    save_image(mask, cfg['predict_path'] + "{}/mask/".format(dotaloader_des), namelist, mode=mode, save_mask=True, plt=tiff)
                    if mode == 'predict':
                        save_image(regress, cfg['predict_path'] + "{}/regress/".format(dotaloader_des), namelist, mode=mode, plt=plt)
                    elif mode == 'train':
                        save_image(regress, cfg['predict_path'] + "{}/regress/".format(dotaloader_des), namelist, mode=mode)
                loss = loss_function(mask, mask_target, regress, regress_target)
                valid_loss = valid_loss + loss.item()
                metrics_value = metric_function(mask, mask_target, regress, regress_target)
                epoch_metrics = record_epoch_metric(epoch_metrics, metrics_value)
                valid_metric = valid_metric + metrics_value[metric_choose]

        epoch_metrics['loss'] = valid_loss
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= (i + 1)
            log.addRecord('valid_' + k, epoch_metrics[k])
            log.drawRecord()
        log.add_scalars('train/all_valid_metrics', epoch_metrics, epoch)
        log.writeLog("  val loss         - {:.6}\n".format(valid_loss))
        log.writeLog("  val metric       - {:.6}\n".format(valid_metric))
        log.writeLog("  valid all metric - {}\n".format(epoch_metrics))
        log.writeLog("_______________________________________________________________\n")

    return valid_loss, valid_metric


def main():
    if args.mode == 'predict':
        log.writeLog("start predict")
        valid(valid_dataloader=train_dataloader, epoch=0, dotaloader_des='train')
        valid(valid_dataloader=valid_dataloader, epoch=0)
    elif args.mode == 'train':
        train(cfg['epoch'])
        drawLogFile(cfg['log_path'])
    elif args.mode == 'test':
        train(cfg['epoch'])
    else:
        print("mode error")


if __name__ == '__main__':
    main()
