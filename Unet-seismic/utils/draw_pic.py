import matplotlib
from matplotlib import pyplot as plt
import json
import numpy as np

matplotlib.style.use('ggplot')

def drawLogFile(logFile_Path):
    logfile = open(logFile_Path + "logFile.txt",'r').readlines()

    train_metric = [each for each in logfile if "train all metric" in each]
    valid_metric = [each for each in logfile if "valid all metric" in each]

    train_metric = [each.split(" - ")[1].strip().replace("\'", "\"") for each in train_metric]
    valid_metric = [each.split(" - ")[1].strip().replace("\'", "\"") for each in valid_metric]

    train_metric = [json.loads(each) for each in train_metric]
    valid_metric = [json.loads(each) for each in valid_metric]

    drawRecord(logFile_Path, train_metric, prefix="train")
    drawRecord(logFile_Path, valid_metric, prefix="valid")


def drawRecord(logFile_Path, train_metric, prefix=""):
    filter_keys = ['tp', 'tn', 'fp', 'fn']
    metrics = {}
    for each in train_metric:
        for key in each.keys():
            if key in filter_keys:
                continue
            metrics[key] = []
            
    for each in train_metric:
        for key in each.keys():
            if key in filter_keys:
                continue
            metrics[key].append(each[key])
    
    History = metrics
    for key in History.keys():
         if key in filter_keys:
             continue
         y_data = History[key]
         y_data = np.array(y_data)
         min = y_data.min()
         max = y_data.max()
         x_range = np.arange(1, len(y_data) + 1)
         fig, ax = plt.subplots(figsize=(40,10))
         ax.plot(x_range, y_data, label=key)
         ax.set_xticks(x_range)
         ax.set_ylim(min - 0.01, max + 0.01)
         # ax.set_yticks(y_range)
         for i, data in enumerate(y_data):
             ax.text(i + 1, data, "{:.4}%".format(data * 100), ha='center', fontsize=5)
         ax.legend()
         fig.savefig(logFile_Path + prefix + "_" + key + ".png",dpi=100)
         plt.close()


def draw_pic(train_metric):
    metrics = {}
    filter_keys = ['tp', 'tn', 'fp', 'fn']
    for each in train_metric:
        for key in each.keys():
            if key in filter_keys:
                continue
            metrics[key] = []
            
    for each in train_metric:
        for key in each.keys():
            if key in filter_keys:
                continue
            metrics[key].append(each[key])
    
    fig, ax = plt.subplots(len(metrics), 1)    
    n = 0
    for key in metrics.keys():
        ax[n].plot(range(1, len(metrics[key]) + 1),metrics[key], label=str(key))
        ax[n].legend()
        n += 1

if __name__ == '__main__':
    pass
    # drawRecord(train_metric, prefix="train")
    # drawRecord(valid_metric, prefix="valid")
