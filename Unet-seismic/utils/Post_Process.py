import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread,imsave
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd

def generateImage(path,path_post):
    center_size = 128
    # path = r"D:\LinuxData\MY_EXPERIMENT\GCResNet\Test_Predict\01\\"
    # path_post = r"D:\LinuxData\MY_EXPERIMENT\GCResNet\Test_Predict\01_post\\"
    files = os.listdir(path)
    masks = {}
    for each in files:
        name = each[-16:]
        masks[name] = np.zeros((1792,1792),np.uint8)
    for each in files:
        x, y = each.split("_")[0], each.split("_")[1]
        x, y = int(x), int(y)
        name = each[-16:]
        image = imread(path + each)
        center = image[center_size//2:center_size//2+center_size, center_size//2:center_size//2+center_size]
        masks[name][x + center_size//2: x + center_size//2 + center_size, 
                    y + center_size//2: y + center_size//2 + center_size] = center 
    for key in masks.keys():
        img = masks[key][146:146+1500,146:146+1500]
        imsave(path_post+key,img)
        
def save_Vision_result(vision_result,vision_path):
    for each in vision_result.keys():
        imsave(vision_path + "/vis_" + each + ".tiff",vision_result[each])
    print("保存到%s完成"%(vision_path))
    
    
def get_one_estimates(predict,gt):
    """
    计算单张图片的各项指标
    """
    eps = 1e-9
    predict = np.asarray(predict)
    gt = np.asarray(gt)
    predict = predict / 255
    gt = gt / 255
    gt = gt.flatten()
    predict = predict.flatten()
    #对gt,pre取非
    repre = np.abs(predict - 1)
    regt = np.abs(gt - 1)
    tp = np.dot(predict,gt.T) + eps#gt1和pre1中均为1的元素数量 P_11
    tn = np.dot(repre,regt.T) + eps#gt1和pre1中均为0的元素数量 P_00
    fp = np.dot(predict,regt.T) + eps#gt1中为0，pre1中为1的元素数量 P_01
    fn = np.dot(repre,gt.T) + eps#gt1中为1，pre1中为0的元素数量 P_10    

    pixel_acc = (tp + tn)/(tp + tn + fp +fn)
    precise = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = (2 * precise * recall)/(precise + recall)
    iou = tp/(tp + fn + fp)
    res = {}
    res['p_acc'] = pixel_acc
    res['precise'] = precise
    res['iou'] = iou
    res['recall'] = recall
    res['f1'] = f1
    return res    

def save_csv(content,save_path):
    cols = ['p_acc','precise','iou','recall','f1']
    df = pd.DataFrame(columns=cols)
    for each in content.keys():
        estimates = content[each]
        df.loc[each] = [estimates['p_acc'],estimates['precise'],estimates['iou'],estimates['recall'],estimates['f1']]
    df.loc['avg'] = df.mean()
    df.to_csv(save_path)

def get_ten_estimates(predict_labels,gts,save_Path):
    """
    :param predict_labels: 存有10张图片的predict label的字典
    :param gts: 存有10张图片的gt的字典
    :return:返回存有10张图片的评价指标的字典
    """
    res = {}
    for each in predict_labels.keys():
        print("生成图片%s的指标"%(each))
        res[each] = get_one_estimates(predict_labels[each],gts[each])
    save_csv(res,save_Path)
    return res

def get_one_vision_result(predict,gt,image):
    """
    :param predict:
    :param gt:
    :param image:
    :return: 染色的image用于保存
    """
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if predict[i][j] == gt[i][j]:
                if predict[i][j] == 255:
                    #TP为绿色
#                    image[i][j][0] = 0
                    image[i][j][1] = min(image[i][j][1] + 75,255)
#                    image[i][j][2] = 0
            elif predict[i][j] == 255:
                #FP为红色
                image[i][j][0] = min(image[i][j][0] + 75,255)
#                image[i][j][1] = 0
#                image[i][j][2] = 0
            else:
                #FN为蓝色
#                image[i][j][0] = 0
#                image[i][j][1] = 0
                image[i][j][2] = min(image[i][j][2] + 75,255)
    return image

def get_ten_vision_result(predict_labels,gts,images):
    res = {}
    for each in predict_labels.keys():
        print("生成图片%s的可视化" % (each))
        res[each] = get_one_vision_result(predict_labels[each],gts[each],images[each])
    return res


gt_path = r"D:\LinuxData\NEW\DataSet\test\gt\\"
img_path = r"D:\LinuxData\mass_buildings\raw\test\sat\\"
pred_path = r"D:\LinuxData\MY_EXPERIMENT\GCResNet\Test_Predict\09_post\\"
path = "D:\\LinuxData\\MY_EXPERIMENT\\GCResNet\\Test_Predict\\09\\"


def getResult(gt_path,img_path,pred_path):
    gt_files = os.listdir(gt_path)
    gt_names = [x.split(".")[0] for x in gt_files]
    print(gt_files)
    gts = {}
    img = {}
    pre = {}
    for each in gt_names:
        gts[each] = imread(gt_path + each + ".tif")
        img[each] = imread(img_path + each + ".tiff")
        pre[each] = imread(pred_path + each + ".tiff")
    res = get_ten_vision_result(pre, gts, img)
    get_ten_estimates(pre, gts, pred_path + "res.csv")
    save_Vision_result(res ,pred_path)
    
if __name__ == '__main__':
    generateImage(path,path_post=pred_path)
    getResult(gt_path, img_path, pred_path)
    
    
    