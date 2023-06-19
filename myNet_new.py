import os
import numpy as np
import random
import torch.nn.functional as F
import torch
import torch.utils.data as dataf
import torch.nn as nn
from scipy import io
import HyperX
import losses
import metric
import datetime
from network import CLMA_Net, GMA_Net


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
# setup_seed(20)
code_start_time = datetime.datetime.now()  # 开始时间点
print("程序运行开始时间：", code_start_time)

"""-------------------------设置参数-------------------------------"""
"""-------------------------PaviaU-------------------------------"""
DataPath = './Data/PaviaU/PaviaU.mat'
TRPath = './Data/PaviaU/PaviaU_eachClass_20_train_1.mat'  # TRPath = './Data/PaviaU/PaviaU_eachClass_20_train_9.mat'
TSPath = './Data/PaviaU/PaviaU_eachClass_20_test_1.mat'  # TSPath = './Data/PaviaU/PaviaU_eachClass_20_test_9.mat'
"""---------------------------KSC-------------------------------"""
# DataPath = './Data/KSC/KSC.mat'
# TRPath = './Data/KSC/KSC_eachClass_20_train_1.mat'
# TSPath = './Data/KSC/KSC_eachClass_20_test_1.mat'
"""---------------------------Houston-------------------------------"""
# DataPath = './Data/Houston/Houston.mat'
# TRPath = './Data/Houston/Houston_eachClass_20_train_1.mat'
# TSPath = './Data/Houston/Houston_eachClass_20_test_1.mat'
"""-------------------------Indian_pines-------------------------------"""
# DataPath = './Data/IndianPines/Indian_pines_corrected.mat'
# TRPath = './Data/IndianPines/IndianPines_eachClass_20_train_5.mat'
# TSPath = './Data/IndianPines/IndianPines_eachClass_20_test_5.mat'
"""-------------------------Salinas--------------------- ----------"""
# DataPath = './Data/salinas/Salinas_corrected.mat'
# TRPath = './Data/salinas/salinas_eachClass_20_train.mat'
# TSPath = './Data/salinas/salinas_eachClass_20_test.mat'


patchsize = 30  # 30  # 例如该值取24，则patch块最后的形状是25×25 [19,23,27,31,35]
batchsize = 32  # 64
EPOCH = 200
LR = 0.0001
"""------------------------设置参数结束-----------------------------"""

# load data
Data = io.loadmat(DataPath)
TrLabel = io.loadmat(TRPath)
TsLabel = io.loadmat(TSPath)

Data = Data['paviaU']
# Data = Data['img']  # Houston 数据集
# Data = Data['salinas_corrected']
# Data = Data['indian_pines_corrected']
# Data = Data['KSC']
# Data = Data['Botswana']
Data = Data.astype(np.float32)
TrLabel = TrLabel['data']
TsLabel = TsLabel['data']

pad_width = np.floor(patchsize / 2)
pad_width = np.int(pad_width)
[m, n, l] = np.shape(Data)  # m=610 n=340 l=103
class_number = np.max(TsLabel)

# 数据归一化
for i in range(l):
    Data[:, :, i] = (Data[:, :, i] - Data[:, :, i].min()) / (Data[:, :, i].max() - Data[:, :, i].min())
x = Data

# 数据边界填充，准备分割数据块
temp = x[:, :, 0]
pad_width = np.floor(patchsize / 2)
pad_width = np.int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2, n2] = temp2.shape
x2 = np.empty((m2, n2, l), dtype='float32')  # 待填充

for i in range(l):
    temp = x[:, :, i]
    pad_width = np.floor(patchsize / 2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x2[:, :, i] = temp2  # x2 为根据patch size四周填充后 patch size=30时，[610,340,103]->[640,370,103,]
# 总共42776 训练180 测试42596
# 构建测试数据集
[ind1, ind2] = np.where(TsLabel != 0)  # 得到不为0的值的的坐标
TestNum = len(ind1)
TestPatch = np.empty((TestNum, l, patchsize + 1, patchsize + 1), dtype='float32')  # (42596,103,31,31)
TestLabel = np.empty(TestNum)  # (42596)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    patch = np.reshape(patch, ((patchsize + 1) * (patchsize + 1), l))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (l, (patchsize + 1), (patchsize + 1)))
    TestPatch[i, :, :, :] = patch
    patchlabel = TsLabel[ind1[i], ind2[i]]
    TestLabel[i] = patchlabel
# TestPatch:[42596,103,30,30], TestLabel:(42596,)
# # 数据集格式转换为tensor
train_dataset = HyperX.dataLoad(x2, TrLabel, patch_size=patchsize, center_pixel=True, flip_augmentation=True)
train_loader = dataf.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

TestPatch = torch.from_numpy(TestPatch)
TestLabel = torch.from_numpy(TestLabel) - 1
TestLabel = TestLabel.long()
Classes = len(np.unique(TrLabel)) - 1

cnn = GMA_Net(classes=class_number, HSI_Data_Shape_H=m, HSI_Data_Shape_W=n, HSI_Data_Shape_C=l,
              patch_size=patchsize + 1)

cnn.cuda()

total = sum([param.nelement() for param in cnn.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_fun1 = losses.ConstractiveLoss()  # the target label is not one-hotted
loss_fun2 = nn.CrossEntropyLoss()
show_feature_map = []
BestAcc = 0
# train and test the designed model
# start training
start_training_time = datetime.datetime.now()

for epoch in range(EPOCH):
    for step, (images, points, labels) in enumerate(
            train_loader):  # gives batch data, normalize x when iterate train_loader

        # move train data to GPU
        images = images.cuda()  # 输入
        points = points.cuda()
        labels = labels.cuda()  # 标签
        bsz = labels.shape[0]

        features3, output = cnn(images, points)  # fake_img:重构结果，output:分类结果

        # 对比损失
        # contrastive_loss = loss_fun1(features3, labels)
        # contrastive_loss = loss_fun1(s_cat, s_label_cat)
        # 分类损失
        classifier_loss = loss_fun2(output, labels)  # 交叉熵，分类误差
        # 总损失
        total_loss = classifier_loss
        # print("epoch",epoch)
        # print("step",step)
        # print("classifier_loss", classifier_loss.data.cpu().numpy())
        # print("contrastive_loss", contrastive_loss.data.cpu().numpy())

        # total_loss = classifier_loss + contrastive_loss

        cnn.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:  # 迭代50次，测试
            cnn.eval()
            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 100
            # 测试为100个bach
            for i in range(number):
                temp = TestPatch[i * 100:(i + 1) * 100, :, :, :]
                temp_points = temp[:, :, pad_width, pad_width]
                temp = temp.cuda()
                temp_points = temp_points.cuda()
                _, temp2 = cnn(temp, temp_points)
                # _,_, temp2 = cnn(temp, temp_points)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
                del temp, temp2, temp3, _, temp_points
            # 不足100个的情况
            if (i + 1) * 100 < len(TestLabel):
                temp = TestPatch[(i + 1) * 100:len(TestLabel), :, :, :]
                temp_points = temp[:, :, pad_width, pad_width]
                temp_points = temp_points.cuda()
                temp = temp.cuda()
                _, temp2 = cnn(temp, temp_points)
                # _, _, temp2 = cnn(temp, temp_points)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
                del temp, temp2, temp3, _, temp_points

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
            # print('Epoch: ', epoch, '| classify loss: %.6f' % classifier_loss.data.cpu().numpy(),
            #       '| contrastive loss: %.6f' % contrastive_loss.data.cpu().numpy(),
            #       '| test accuracy（OA）: %.6f' % accuracy)
            print('Epoch: ', epoch, '| classify loss: %.6f' % classifier_loss.data.cpu().numpy(),
                  '| test accuracy（OA）: %.6f' % accuracy)

            # save the parameters in network
            if accuracy > BestAcc:
                # torch.save(cnn.state_dict(), 'net_params_myNet_HS.pkl')
                torch.save(cnn.state_dict(), 'net_params_myNet_UP.pkl')
                BestAcc = accuracy
            cnn.train()

# end training time
end_training_time = datetime.datetime.now()

# # test each class accuracy
cnn.load_state_dict(torch.load('net_params_myNet_UP.pkl'))
cnn.eval()

# start testing time
start_testing_time = datetime.datetime.now()

pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel) // 100
for i in range(number):
    temp = TestPatch[i * 100:(i + 1) * 100, :, :, :]
    temp_points = temp[:, :, pad_width, pad_width]
    temp = temp.cuda()
    temp_points = temp_points.cuda()
    _, temp2 = cnn(temp, temp_points)
    # _, _, temp2 = cnn(temp, temp_points)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
    del temp, temp2, temp3, _, temp_points
# 不足100个的情况
if (i + 1) * 100 < len(TestLabel):
    temp = TestPatch[(i + 1) * 100:len(TestLabel), :, :, :]
    temp_points = temp[:, :, pad_width, pad_width]
    temp_points = temp_points.cuda()
    temp = temp.cuda()
    _, temp2 = cnn(temp, temp_points)
    # _, _, temp2 = cnn(temp, temp_points)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
    del temp, temp2, temp3, _, temp_points

# end testing time
end_testing_time = datetime.datetime.now()

# 评价指标
pred_y = torch.from_numpy(pred_y).long()
Classes = np.unique(TestLabel)
EachAcc = np.empty(len(Classes))
AA = 0.0
for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0
    for j in range(len(TestLabel)):
        if TestLabel[j] == cla:
            sum += 1
        if TestLabel[j] == cla and pred_y[j] == cla:
            right += 1
    EachAcc[i] = right.__float__() / sum.__float__()
    AA += EachAcc[i]

print('-------------------')
for i in range(len(EachAcc)):
    # print('|第%d类精度：' % (i + 1), '%.2f|' % (EachAcc[i] * 100))
    print('%.2f' % (EachAcc[i] * 100))
    # print('-------------------')
AA *= 100 / len(Classes)

results = metric.metrics(pred_y, TestLabel, n_classes=len(Classes))
# print('test accuracy（OA）: %.2f ' % results["Accuracy"], 'AA : %.2f ' % AA, 'Kappa : %.2f ' % results["Kappa"])
print('%.2f' % results["Accuracy"])
print('%.2f' % AA)
print('%.2f' % results["Kappa"])
# print('confusion matrix :')
# print(results["Confusion matrix"])
code_end_time = datetime.datetime.now()  # 结束时间点
print("程序运行结束时间：", code_end_time)
print('程序运行总时长：', code_end_time - code_start_time)  # 运行时间，单位是  时:分:秒
print('训练时长：', end_training_time - start_training_time)  # 运行时间，单位是  时:分:秒
print('测试时长：', end_testing_time - start_testing_time)  # 运行时间，单位是  时:分:秒
