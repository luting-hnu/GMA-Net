import numpy as np
from scipy import io, misc

def split_dataset_equal(gt_2D, numTrain=20):
    """
    功能：按照每类数量分割训练集与测试集
    输入：（二维原始标签y，每类训练数量）
    输出：训练集一维坐标,测试集一维坐标
    """
    gt_1D = np.reshape(gt_2D, (-1, 1))
    train_gt_1D = np.zeros_like(gt_1D)
    test_gt_1D = np.zeros_like(gt_1D)
    train_idx, test_idx, numList = [], [], []
    numClass = np.max(gt_1D)  # 获取最大类别数
    for i in range(1, numClass + 1):  # 忽略背景元素
        idx = np.where(gt_1D == i)[0]  # 记录下该类别的坐标值
        numList.append(len(idx))  # 得到该类别的数量
        np.random.shuffle(idx)  # 对坐标乱序
        if len(idx) < numTrain:
            train_idx.append(idx[:(numTrain // 2)])  # 样本不足，收集每一类的前n/2个作为训练样本
            test_idx.append(idx[numTrain // 2:])  # 收集每一类剩余的作为测试样本
        else:
            train_idx.append(idx[:numTrain])  # 收集每一类的前n个作为训练样本
            test_idx.append(idx[numTrain:])  # 收集每一类剩余的作为测试样本
    for i in range(len(train_idx)):
        for j in range(len(train_idx[i])):
            train_gt_1D[train_idx[i][j]] = i + 1
        for k in range(len(test_idx[i])):
            test_gt_1D[test_idx[i][k]] = i + 1
    train_gt_2D = np.reshape(train_gt_1D, (gt_2D.shape[0], gt_2D.shape[1]))
    test_gt_2D = np.reshape(test_gt_1D, (gt_2D.shape[0], gt_2D.shape[1]))
    return train_gt_2D, test_gt_2D

GTDataPath = '/home/luting/桌面/BaiClassification/MTGAN-main/Data/PaviaU/PaviaU_gt.mat'
GT = io.loadmat(GTDataPath)
# GT = GT['indian_pines_gt10']
# GT = GT['Houston_gt']
# GT = GT['salinas_gt']
GT = GT['paviaU_gt']
# GT = GT['Botswana_gt']
train_gt_2D, test_gt_2D = split_dataset_equal(GT, numTrain=5)
io.savemat('./Data/PaviaU/PaviaU_eachClass_5_train_9.mat', {'data': train_gt_2D})  #保存mat文件
io.savemat('./Data/PaviaU/PaviaU_eachClass_5_test_9.mat', {'data': test_gt_2D})