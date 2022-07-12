import numpy as np

import torch as t

def get_Init_Metrics():
    return {
            'mse':0.0,
            'mae':0.0,
            'rmse':0.0,
            'acc': 0.0,
            'iou': 0.0,
            'f1': 0.0,
            'precise': 0.0,
            'recall': 0.0,
            'tp': 0.0,
            'fp': 0.0,
            'tn': 0.0,
            'fn': 0.0
            }


def get_tp_tn_fp_fn(output, target):
    N = output.shape[0]
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    if output.shape[1] == 2:
        output = output.argmax(axis=1)
    else:
        output = np.where(output > 0.5, 1, 0)

    if target.shape[1] == 2:
        target = target.argmax(axis=1)

    output = output.flatten()
    target = target.flatten()

    re_output = np.abs(output - 1)
    re_target = np.abs(target - 1)

    tp = (output * target).sum() # gt=1,pre=1
    tn = (re_output * re_target).sum() # gt=0,pre=0
    fp = (output * re_target).sum() #gt=0,pre=1
    fn = (re_output * target).sum()#gt=1,pre=0
    return tp, tn, fp, fn

def cal_f1score(tp, tn, fp, fn):
    recall = tp / (tp + fn + 1e-9)
    precise = tp / (tp + fp + 1e-9)
    f1 = (2 * recall * precise) / (recall + precise)

def cal_recall(tp, tn, fp, fn):
    recall = tp /(tp + fn + 1e-9)
    return recall

def cal_precise(tp, tn, fp, fn):
    precise = tp /(tp + fp + 1e-9)
    return precise

def cal_iou(tp, tn, fp, fn):
    iou = tp /(tp + fp + fn + 1e-9)
    return iou

def cal_acc(tp, tn, fp, fn):
    acc = (tp + tn)/(tp + tn + fp + fn)
    return acc


def metric_function(mask, mask_target, regress, regress_target):
    mask = t.sigmoid(mask)
    N = regress.shape
    b = N[0]
    c = N[1]
    h = N[2]
    w = N[3]
    size = b * c * h * w
    mean_abs_error = (t.abs(regress - regress_target).sum() / size).cpu().detach().numpy().mean()
    mean_square_error = (((regress - regress_target) * (regress - regress_target)).sum() / size).cpu().detach().numpy().mean()

    tp, tn, fp, fn = get_tp_tn_fp_fn(mask, mask_target)
    acc = cal_acc(tp, tn, fp, fn)
    iou = cal_iou(tp, tn, fp, fn)
    recall = cal_recall(tp, tn, fp, fn)
    precise = cal_precise(tp, tn, fp, fn)
    f1 = (2 * recall * precise)/(recall + precise + 0.01)

    return {
            'mse': mean_square_error,
            'rmse': np.sqrt(mean_square_error),
            'mae': mean_abs_error,
            'acc': acc,
            'iou': iou,
            'recall': recall,
            'precise': precise,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
            }

def record_epoch_metric(epoch_metrics, metrics_value):
    for k in metrics_value.keys():
        epoch_metrics[k] += metrics_value[k]
    return epoch_metrics

if __name__ == '__main__':
    output = t.tensor([-0.7,0.8,0.1,0.4])
    target = t.tensor([0.0,1.0,1.0,0.0])
    output = output.reshape((1,1,2,2))
    target = target.reshape((1,1,2,2))

