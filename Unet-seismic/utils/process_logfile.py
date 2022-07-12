# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:37:35 2019

@author: hxw
"""
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

lines = open("../log/6.16.15.37.55.baseline_batch_8_bce_DenseNet_4_12_64_with_gc_['channel_add', 'channel_mul']/logFile.txt").readlines()
valid = []
precise = []
iou = []
recall = []
f1 = []
acc =[]
loss = []
for each in lines:
    if 'valid all metric' in each:
        each = each.split("-")[1]
        each = each.replace("\'","\"")
        d = json.loads(each)
        acc.append(d['acc'])
        iou.append(d['iou'])
        f1.append(d['f1'])
        recall.append(d['recall'])
        precise.append(d['precise'])
        loss.append(d['loss'])
        
x_range = np.arange(1,len(acc) + 1)
plt.plot(x_range, acc, label='acc')
plt.plot(x_range, precise, label='precise')
plt.plot(x_range, f1, label='f1')
plt.plot(x_range, recall, label='recall')
plt.plot(x_range, iou, label='iou')
plt.plot(x_range, loss, label='loss')
plt.legend()

