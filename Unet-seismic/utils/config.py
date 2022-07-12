import torch as t
import numpy as np
import random
import torch.nn as nn
# from model.UNet import UNet
from model.UNet_mult_up_path import UNet
from utils.logger import logger
from utils.makedirs import mkdirs
from model.DenseUNet_concat import DenseUNet
from dataset.dataset_processor import get_train_valid_dataloader
from dataset.dataset_processor import SeismicData, SeismicToJPEG, SeismicMultSourceData
from utils.loss import MultLoss

def getExperimentConfig(experiment_name='', mode='train', dataset='simple', batch=32, image_channel=1, base_channel=32):
    config = {}
    config['task_class'] = 'mult'
    config['batch'] = batch
    config['epoch'] = 40
    if dataset == 'simple':
        config['train_path'] = "./dataset/simple_arch/"
        config['valid_path'] = "./dataset/simple_arch/"
    elif dataset == 'complex':
        config['train_path'] = "./dataset/"
        config['valid_path'] = "./dataset/"   
    elif dataset == 'mix':
        config['train_path'] = "./dataset/mix_arch/"
        config['valid_path'] = "./dataset/mix_arch/"
    elif dataset == 'other_paper':
        config['train_path'] = "./dataset/other_paper_arch/"
        config['valid_path'] = "./dataset/other_paper_arch/"     
    else:
        raise ValueError   

    config['dataset_raw_path'] = './dataset/raw_data/'
    if image_channel == 1:
        config['train_dataloader'], config['valid_dataloader']  = get_train_valid_dataloader(config, get_loader=True,
                                                                                             SeismicData=SeismicData)
    elif image_channel == 5:
        config['train_dataloader'], config['valid_dataloader']  = get_train_valid_dataloader(config, get_loader=True,
                                                                                             SeismicData=SeismicMultSourceData)
    else:
        raise ValueError  

    config['model'] = UNet(input_channels=image_channel, class_num=1, base_channel=base_channel)
    # config['model'] = DenseUNet(num_block_layer=4, block_out_channls=64, growth_rate=12, efficient=False, class_num=2,
    #                             use_aspp=False, use_context=False, gc_fusion_method=['channel_mul','channel_add'])
    config['optimizer'] = t.optim.SGD(config['model'].parameters(), lr=0.01, weight_decay=1e-6)
    config['loss_function_seg'] = nn.BCEWithLogitsLoss(pos_weight=t.tensor([1.0, 500.0]).cuda())
    config['loss_function_reg'] = nn.MSELoss()
    if dataset == 'mix' or dataset == 'simple' or dataset == 'complex':
        config['loss_function_mult'] = MultLoss(bceweight=0.0, mseweight=10)
    else:
        config['loss_function_mult'] = MultLoss(bceweight=0.0, mseweight=0)
    config['experiment_name'] = experiment_name
    config['checkpoint_path'] = "./checkpoint/" + experiment_name + "/"
    config['model_checkpoint_name'] = "model.pth"
    config['optimizer_checkpoint_name'] = "optim.pth"
    config['log_path'] = "./log/" + experiment_name + "/"
    config['predict_path'] = './predict/' + experiment_name + "/"
    config['load_pretrain_model'] = False
    config['load_pretrain_optim'] = False
    config['logger'] = logger(config['log_path'], mode=mode)
    if mode == 'train' or mode == 'predict':
        mkdirs(config['log_path'])
        mkdirs(config['predict_path'] + 'valid/')
        mkdirs(config['predict_path'] + 'train/')
    else:
        config['log_path'] = './log/test/'
    return config

def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True

if __name__ == '__main__':
    cfg = getExperimentConfig(experiment_name='FCN', mode='test')