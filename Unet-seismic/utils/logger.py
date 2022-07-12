import matplotlib
import tensorboardX as tfx
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

class logger():
    def __init__(self, log_dir, mode='train'):
        print("\nOutput log events to dir {}\n".format(log_dir))
        self.mode = mode
        self.History = {}
        self.log_dir = log_dir

        if self.mode == 'train' or self.mode == 'predict':
            self.logger = tfx.SummaryWriter(log_dir)
            self.logFile = open(log_dir + "/logFile.txt", 'a')
        else:
            try:
                self.logFile = open('./log/test/' + "logFile.txt", 'a')
            except FileNotFoundError:
                print("WARNING: log file not create")

    def add_scalar(self,table_name,x,y):
        if self.mode == 'train':
            self.logger.add_scalar(table_name,x,y)

    def add_scalars(self,main_tag,tag_scalar_dict, global_step=None):
        if self.mode == 'train':
            self.logger.add_scalars(main_tag, tag_scalar_dict, global_step)

    def printInfo(self, batch,epoch,loss_function,optimizer,use_gpu):
        self.writeLog("Batch Size    : {}\n".format(batch))
        self.writeLog("Train Epoch   : {}\n".format(epoch))
        self.writeLog("Loss Function : {}\n".format(loss_function.__class__))
        self.writeLog("Optimizer     : {}\n".format(optimizer.__class__))
        self.writeLog("Use GPU       : {}\n".format(use_gpu))

    def writeLog(self, msg):
        self.logFile.writelines(msg)
        self.logFile.flush()

    def drawRecord(self):
        return
        # for key in self.History.keys():
        #     y_data = self.History[key]
        #     y_data = np.array(y_data)
        #     min = y_data.min()
        #     max = y_data.max()
        #     x_range = np.arange(1, len(y_data) + 1)
        #     fig, ax = plt.subplots(figsize=(16,10))
        #     ax.plot(x_range, y_data, label=key)
        #     ax.set_xticks(x_range)
        #     ax.set_ylim(min - 0.01, max + 0.01)
        #     # ax.set_yticks(y_range)
        #     for i, data in enumerate(y_data):
        #         ax.text(i + 1, data, "{:.4}%".format(data * 100), ha='center', fontsize=12)
        #     ax.legend()
        #     fig.savefig(self.log_dir + '/' + key + ".png",dpi=100)
        #     plt.close()


    def addRecord(self, key, value):
        """
        记录随epoch变化的一些指标
        """
        try:
            self.History[key].append(value)
        except KeyError:
            self.History[key] = []
            self.History[key].append(value)


if __name__ == '__main__':
    log = logger(log_dir='./', mode='test')
    for i in range(40):
        log.addRecord('acc',np.random.random())
    log.drawRecord()
    # for i in range(10):
    #     log.addRecord('acc',i + np.random.random() * 6 - 3)
    # log.drawRecord()