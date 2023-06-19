import numpy as np
import torch
import itertools
from tqdm import tqdm
from scipy import io
# import h5py
# # Visualization
# import seaborn as sns
# import visdom
import torch.nn as nn
import torch.nn.functional as F
# from utils import convert_to_color_, convert_from_color_
import matplotlib.pyplot as plt


"""-----------------------------------网络模型---------------------------------------"""
"""------------------------------------CLMA-----------------------------------------"""
#模型
class ChannelAttention_LliuMK3(nn.Module):
    def __init__(self, in_planes, rotio=2):
        super(ChannelAttention_LliuMK3, self).__init__()
        self.rotio = rotio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // self.rotio, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // self.rotio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        out = self.coefficient1 * avg_out + self.coefficient2 * max_out
        return self.sigmoid(out)

class MultiChannelAttention_LliuMK3(nn.Module):
    def __init__(self, in_planes, rotio=2):
        super(MultiChannelAttention_LliuMK3, self).__init__()
        self.in_planes = in_planes
        self.rotio = rotio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, (1, 1), bias=False))

        self.conv = nn.Conv2d(in_channels=self.in_planes, out_channels=self.in_planes, kernel_size=(1, 1),
                              stride=(1, 1), padding=(0, 0))
        self.batch_norm = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU()
        # self.up_sample_fc = nn.Sequential(nn.Linear(in_features=in_planes, out_features=in_planes),
        #                                   nn.BatchNorm1d(num_features=in_planes),
        #                                   nn.ReLU()
        #                                   )
        self.sigmoid = nn.Sigmoid()

        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x, shallow_channel_attention_map):  # 64,103,3,3   64,103,1,1
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))

        # x2 = shallow_channel_attention_map.squeeze(-1).squeeze(-1)
        # x2 = self.up_sample_fc(x2)
        # x2 = x2.unsqueeze(-1).unsqueeze(-1)
        # x2 = shallow_channel_attention_map.permute(0, 2, 1, 3)  # 64,103,1,1 -> 64,1,103,1
        x2 = shallow_channel_attention_map  # 64,103,1,1 -> 64,1,103,1

        x2 = self.conv(x2)
        x2 = self.batch_norm(x2)
        x2 = self.relu(x2)
        # x2 = x2.permute(0, 2, 1, 3)
        x2 = self.sharedMLP(x2)
        out = self.coefficient1 * avg_out + self.coefficient2 * max_out + self.coefficient3 * x2
        return self.sigmoid(out)


class SpatialAttention_LliuMK3(nn.Module):
    def __init__(self):
        super(SpatialAttention_LliuMK3, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [16,32,9,9]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [16, 1, 9, 9]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [16, 1, 9, 9]
        x_out = torch.cat([avg_out, max_out], dim=1)  # [16,2,9,9]                                    # 按维数1（列）拼接
        x_out = self.conv(x_out)
        return self.sigmoid(x_out)


class MultiSpatialAttention_LliuMK3(nn.Module):
    def __init__(self):
        super(MultiSpatialAttention_LliuMK3, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.batch_norm0 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, x, shallow_spatial_attention_map):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # shallow_spatial_attention_map = nn.MaxPool2d(2)(shallow_spatial_attention_map)

        x0 = self.conv0(shallow_spatial_attention_map)
        x0 = self.batch_norm0(x0)
        x0 = self.relu(x0)
        x0 = nn.AvgPool2d(2)(x0)  # 最大池化下采样效果貌似稍微好一点，还需要进一步实验验证
        x = torch.cat([avg_out, max_out, x0], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class LliuMK_Net3(nn.Module):
    def __init__(self, classes, HSI_Data_Shape_H, HSI_Data_Shape_W, HSI_Data_Shape_C):  # band:103  classes=9
        super(LliuMK_Net3, self).__init__()
        self.name = 'LliuMK_Net3'
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C

        # self.mish = mish()  # 也可以引用一下，等待后续改进
        self.relu = nn.ReLU()

        self.CA1 = ChannelAttention_LliuMK3(in_planes=self.band)
        # self.CA2 = ChannelAttention_LliuMK(in_planes=64)
        # self.CA3 = ChannelAttention_LliuMK(in_planes=128)
        # self.CA4 = ChannelAttention_LliuMK(in_planes=256)

        self.MCA1 = MultiChannelAttention_LliuMK3(in_planes=self.band)
        self.MCA2 = MultiChannelAttention_LliuMK3(in_planes=self.band)
        # self.MCA3 = MultiChannelAttention_LliuMK(256)

        self.SA1 = SpatialAttention_LliuMK3()
        # self.SA2 = SpatialAttention_LliuMK()
        # self.SA3 = SpatialAttention_LliuMK()
        # self.SA4 = SpatialAttention_LliuMK()
        # self.CA21 = ChannelAttention_for_spatial_LliuMK(in_planes=32)
        # self.MCA21 = MultiChannelAttention_for_spatial_LliuMK(64)
        # self.MCA22 = MultiChannelAttention_for_spatial_LliuMK(128)

        self.MSA1 = MultiSpatialAttention_LliuMK3()
        self.MSA2 = MultiSpatialAttention_LliuMK3()
        # self.MSA3 = MultiSpatialAttention_LliuMK()
        # self.MSA4 = MultiSpatialAttention_LliuMK()

        # self.Ca = ChannelAttention(in_planes=24)
        # self.MCa1 = MultiChannelAttention(in_planes=24)
        # self.MCa2 = MultiChannelAttention(in_planes=24)
        # self.MCa3 = MultiChannelAttention(in_planes=24)
        #
        # self.Sa = SpatialAttention()
        # self.MSa1 = MultiSpatialAttention()
        # self.MSa2 = MultiSpatialAttention()
        # self.MSa3 = MultiSpatialAttention()

        # self.lbp = LBP()
        # self.dense_block = DenseBlock()

        # spectral branch
        # self.FC11 = nn.Linear(self.band, 32)
        # self.conv11 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(7, 1),
        #                         stride=(1, 1), padding=(3, 0))
        # self.batch_norm11 = nn.BatchNorm2d(3)
        self.conv11 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm11 = nn.BatchNorm2d(self.band)

        # self.FC12 = nn.Linear(32, 64)
        # self.conv12 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(7, 1),
        #                         stride=(1, 1), padding=(3, 0))
        # self.batch_norm12 = nn.BatchNorm2d(3)
        self.conv12 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm12 = nn.BatchNorm2d(self.band)

        # self.FC13 = nn.Linear(64, 128)
        # self.conv13 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(7, 1),
        #                         stride=(1, 1), padding=(3, 0))
        # self.batch_norm13 = nn.BatchNorm2d(3)
        self.conv13 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm13 = nn.BatchNorm2d(self.band)

        # self.FC14 = nn.Linear(128, 256)
        # self.conv14 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1),
        #                         stride=(1, 1), padding=(0, 0))

        # self.batch_norm14 = nn.BatchNorm2d(256)

        # self.FC15 = nn.Linear(64, 128)
        # self.conv15 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.batch_norm15 = nn.BatchNorm1d(128)

        # self.fc_spectral = nn.Linear(256, self.classes)

        # Spatial Branch
        self.conv21 = nn.Conv2d(in_channels=self.band, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
        self.batch_norm21 = nn.BatchNorm2d(64)

        self.conv22 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch_norm22 = nn.BatchNorm2d(128)

        self.conv23 = nn.Conv2d(in_channels=128, out_channels=self.band, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
        self.batch_norm23 = nn.BatchNorm2d(self.band)

        # self.conv24 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.batch_norm24 = nn.BatchNorm2d(256, affine=True)
        #
        # self.conv25 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1))
        # self.batch_norm25 = nn.BatchNorm2d(256, affine=True)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.fc_spatial = nn.Linear(256, self.classes)

        # Texture Branch
        # self.pca_lbp_for_LliuMK_Net3 = PCA_LBP_for_LliuMK_Net3(pca_num=2)
        # self.conv31 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.batch_norm31 = nn.BatchNorm2d(32, affine=True)
        #
        # self.conv32 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.batch_norm32 = nn.BatchNorm2d(64, affine=True)
        #
        # self.conv33 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.batch_norm33 = nn.BatchNorm2d(128, affine=True)

        # self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.5]))
        # self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.5]))

        self.finally_fc_classification = nn.Linear(self.band * 2, self.classes)

    def forward(self, patchX, pixelX):  # x:(16,103,9,9)
        """------------------------光谱分支------------------------"""
        patch_size = patchX.shape[-1] // 2
        input_spectral = patchX[:, :, (patch_size-1):(patch_size + 2), (patch_size-1):(patch_size + 2)]  # [64,103,3,3]

        # input_spectral = input_spectral.permute(0, 2, 1, 3)  # [64,3,103,3]
        x11 = self.conv11(input_spectral)
        x11 = self.batch_norm11(x11)  # [64,103,3,3]
        x11 = self.relu(x11)
        # x11 = x11.permute(0, 2, 1, 3)
        ca1 = self.CA1(x11)
        # x11 = torch.cat((x11, x11*ca1), dim=1)
        x11 = x11 * ca1

        # x12 = x11.permute(0, 2, 1, 3)
        x12 = x11
        x12 = self.conv12(x12)
        x12 = self.batch_norm12(x12)
        x12 = self.relu(x12)
        # x12 = x12.permute(0, 2, 1, 3)
        # ca2 = self.CA2(x12)
        # x12 = x12 * ca2
        mca1 = self.MCA1(x12, ca1)
        # x12 = torch.cat((x12, x12 * mca1), dim=1)
        x12 = x12 * mca1

        # x13 = x12.permute(0, 2, 1, 3)
        x13 = x12
        x13 = self.conv13(x13)
        x13 = self.batch_norm13(x13)
        x13 = self.relu(x13)
        # x13 = x13.permute(0, 2, 1, 3)
        # ca3 = self.CA3(x13)
        # x13 = x13 * ca3
        mca2 = self.MCA2(x13, mca1)
        # x13 = torch.cat((x13, x13 * mca2), dim=1)
        x13 = x13 * mca2

        # x14 = self.FC14(x13)
        # x14 = self.batch_norm14(x14)
        # # ca4 = self.CA4(x14)
        # # x14 = x14 * ca4
        # mca3 = self.MCA3(x14, mca2)
        # x14 = x14 * mca3
        # x14 = self.relu(x14)
        #
        x13 = self.global_pooling(x13)
        x13 = x13.view(x13.size(0), -1)
        output_spectral = x13

        # output_spectral = self.fc_spectral(x14)
        # output_spectral = F.softmax(output_spectral, dim=1)

        """------------------------空间分支------------------------"""
        input_spatial = patchX
        x21 = self.conv21(input_spatial)  # (16,32,9,9)<—(16,103,9,9)
        x21 = self.batch_norm21(x21)  # (16,32,9,9)
        x21 = self.relu(x21)  # (16,32,9,9)
        # x21 = x21 * ca1.unsqueeze(-1).unsqueeze(-1)
        # ca21 = self.CA21(x21)
        # x21 = x21 * ca21
        sa1 = self.SA1(x21)
        # x21 = torch.cat((x21, x21 * sa1), dim=1)
        x21 = x21 * sa1
        x21 = nn.MaxPool2d(2)(x21)

        x22 = self.conv22(x21)  # (16,24,1,9,9)
        x22 = self.batch_norm22(x22)  # (16,24,1,9,9)
        x22 = self.relu(x22)
        # sa2 = self.Spatial_Attention_2(x22)
        # x22 = x22 * sa2
        # x22 = x22 * mca1.unsqueeze(-1).unsqueeze(-1)
        # mca21 = self.MCA21(x22, ca21)
        # x22 = x22 * mca21
        msa1 = self.MSA1(x22, sa1)
        # x22 = torch.cat((x22, x22 * msa1), dim=1)
        x22 = x22 * msa1

        x22 = nn.MaxPool2d(2)(x22)

        x23 = self.conv23(x22)  # (16,24,1,9,9)
        x23 = self.batch_norm23(x23)  # (16,24,1,9,9)
        x23 = self.relu(x23)
        # sa3 = self.Spatial_Attention_3(x23)
        # x23 = x23 * sa3
        # x23 = x23 * mca2.unsqueeze(-1).unsqueeze(-1)
        # mca22 = self.MCA22(x23, mca21)
        # x23 = x23 * mca22
        msa2 = self.MSA2(x23, msa1)
        # x23 = torch.cat((x23, x23 * msa2), dim=1)
        x23 = x23 * msa2

        x23 = nn.MaxPool2d(2)(x23)

        # x24 = self.conv24(x23)  # (16,24,1,9,9)
        # x24 = self.batch_norm24(x24)  # x16:(16,24,9,9,1)
        # # x24 = nn.MaxPool2d(2)(x24)
        # # sa4 = self.Spatial_Attention_4(x24)
        # # x24 = x24 * sa4
        # x24 = x24 * mca3.unsqueeze(-1).unsqueeze(-1)
        # msa3 = self.MSA3(x24, msa2)
        # x24 = x24 * msa3
        # x24 = self.relu(x24)

        x25 = self.global_pooling(x23)
        x25 = x25.view(x25.size(0), -1)
        output_spatial = x25

        output = torch.cat((output_spectral, output_spatial), dim=1)
        output = self.finally_fc_classification(output)
        output = F.softmax(output, dim=1)

        return output, output



"""----------------------------------网络模型结束------------------------------------"""
# Net_Name = 'net_params_myNet_HS_PaviaU_20.pkl'  # 样本1
# Net_Name = 'net_params_myNet_HS_PaviaU_20_3.pkl'  # 样本2
Net_Name = 'net_params_myNet_UP.pkl'  # 样本2
patchsize = 22
batchsize = 200


# DataPath = './Data/Houston/Houston.mat'
# GTPath = './Data/Houston/Houston_gt.mat'
# Data = io.loadmat(DataPath)
# GT = io.loadmat(GTPath)
# Data = Data['img']
# GT = GT['Houston_gt']
# class_number = 15


# DataPath = './Data/PaviaU/PaviaU.mat'
# GTPath = './Data/PaviaU/PaviaU_gt.mat'
# Data = io.loadmat(DataPath)
# GT = io.loadmat(GTPath)
# Data = Data['paviaU']
# GT = GT['paviaU_gt']
# class_number = 9
# Classes=9

DataPath = './Data/salinas/Salinas_corrected.mat'
GTPath = './Data/salinas/Salinas_gt.mat'
Data = io.loadmat(DataPath)
GT = io.loadmat(GTPath)
Data = Data['salinas_corrected']
GT = GT['salinas_gt']
class_number = 16

# DataPath = './Data/IndianPines/Indian_pines_corrected.mat'
# GTPath = './Data/IndianPines/Indian_pines_gt10.mat'
# Data = io.loadmat(DataPath)
# GT = io.loadmat(GTPath)
# Data = Data['indian_pines_corrected']
# GT = GT['indian_pines_gt10']
# class_number = 10




# #

# DataPath = './Data/PaviaU/PaviaU.mat'
# IMG = h5py.File(DataPath)               #读取mat文件
# Data = IMG['HSI'][:]
# Data = IMG['HSI'][:]
# Data = np.transpose(Data, (1, 2, 0))
# GT = IMG['GT'][:]
# DataPath = './Data/IndianPines/Indian_pines_corrected.mat'
# GTPath = './Data/IndianPines/Indian_pines_gt10.mat'

# #
# DataPath = './Data/KSC/KSC.mat'
# # GTPath = './Data/KSC/KSC_gt.mat'


# # Data = Data['indian_pines_corrected']


# Data = Data['salinas_corrected']
# Data = Data['KSC']
Data = Data.astype(np.float32)
# GT = GT['KSC_gt']


# GT = GT['indian_pines_gt10']
[m, n, l] = np.shape(Data)

# normalization
for i in range(l):
    Data[:, :, i] = (Data[:, :, i] - Data[:, :, i].min()) / (Data[:, :, i].max() - Data[:, :, i].min())
# 数据边界填充，准备分割数据块
temp = Data[:, :, 0]
pad_width = np.floor(patchsize / 2)
pad_width = np.int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2, n2] = temp2.shape
x2 = np.empty((m2, n2, l), dtype='float32')  # 待填充

for i in range(l):
    temp = Data[:, :, i]
    pad_width = np.floor(patchsize / 2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x2[:, :, i] = temp2  # x2 为根据patch size四周填充后 patch size=30时，[610,340,103]->[640,370,103,]
Data = x2
# 数据边界填充完毕
# load model
# model = torch.load(Net_Name)
# model = LliuMK_MS(classes=class_number, HSI_Data_Shape_H=m, HSI_Data_Shape_W=n, HSI_Data_Shape_C=l,
#                   patch_size=patchsize).cuda()
# model = Network().cuda()
model = LliuMK_Net3(classes=class_number, HSI_Data_Shape_H=m, HSI_Data_Shape_W=n, HSI_Data_Shape_C=l).cuda()

model.load_state_dict(torch.load(Net_Name))

img = Data
gt = GT
# batchsize= 400

Classes = len(np.unique(gt)) - 1  # 10


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    if w % 2 == 0:
        end_H = H - h + offset_h
        end_W = W - w + offset_w
    else:
        end_H = H - h + offset_h + 1
        end_W = W - w + offset_w + 1
    for x in range(0, end_W, step):
        if x + w > W:
            x = W - w
        for y in range(0, end_H, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def euclidean_dist(x, y):
    """输入：[b1, c, w, n], [b2, c, w, n]
           输出：[b1, b2]
        计算两个矩阵中两两欧氏距离"""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def DrawResult(labels, imageID):
    # ID=1:MUUFL
    # ID=2:Indian Pines
    # ID=3: pavia
    # ID=7:Houston
    # ID=8:Houston 18
    # ID=12:NC12
    # ID=13:NC13
    # ID=16:NC16
    global palette
    global row
    global col
    num_class = int(labels.max())
    if imageID == 1:
        row = 325
        col = 220
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212]])
        palette = palette * 1.0 / 255

    elif imageID == 2:
        row = 145
        col = 145
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 3:
        row = 610
        col = 340
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240]])
        palette = palette * 1.0 / 255
    elif imageID == 4:
        row = 166
        col = 600
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255]])
        palette = palette * 1.0 / 255
    elif imageID == 5: # salinas
        row = 512
        col = 217
        palette = np.array([[255,0,0],
                            [0,255,0],
                            [0,0,255],
                            [255,255,0],
                            [0,255,255],
                            [255,0,255],
                            [176,47,95],
                            [45,139,87],
                            [160,31,240],
                            [255,127,80],
                            [127,255,212],
                            [217,112,214],
                            [160,81,44],
                            [127,255,0],
                            [216,191,216],
                            [237,0,0]])
        palette = palette * 1.0 / 255
    elif imageID == 7:
        row = 349
        col = 1905
        # palette = np.array([[53, 71, 157],
        #                     [56, 81, 163],
        #                     [62, 95, 172],
        #                     [69, 147, 209],
        #                     [76, 199, 233],
        #                     [108, 200, 195],
        #                     [130, 200, 137],
        #                     [179, 211, 52],
        #                     [230, 229, 23],
        #                     [232, 189, 32],
        #                     [238, 136, 35],
        #                     [240, 72, 35],
        #                     [238, 41, 35],
        #                     [191, 32, 37],
        #                     [125, 21, 22]])

        # palette = np.array([[255, 0, 0],  # R G B
        #                     [0, 255, 0],
        #                     [0, 0, 255],
        #                     [255, 255, 0],
        #                     [0, 255, 255],
        #                     [255, 127, 80],
        #                     [127, 255, 212],
        #                     [218, 112, 214],
        #                     [160, 82, 45],
        #                     [127, 255, 0],
        #                     [255, 0, 255],
        #                     [176, 48, 96],
        #                     [46, 139, 87],
        #                     [160, 32, 240],
        #                     [216, 191, 216]])

        palette = np.array([[1, 204, 1],
                            [128, 255, 0],
                            [49, 137, 87],
                            [1, 139, 0],
                            [160, 82, 46],
                            [1, 255, 255],
                            [255, 255, 255],
                            [215, 191, 215],
                            [254, 0, 0],
                            [138, 0, 2],
                            [0, 0, 0],
                            [255, 255, 1],
                            [239, 154, 1],
                            [85, 26, 142],
                            [255, 127, 80]])

        palette = palette * 1.0 / 255
    elif imageID == 12:
        row = 682
        col = 2884
        palette = np.array([[53, 71, 157],
                            [56, 81, 163],
                            [62, 95, 172],
                            [69, 147, 209],
                            [76, 199, 233],
                            [108, 200, 195],
                            [130, 200, 137],
                            [179, 211, 52],
                            [230, 229, 23],
                            [232, 189, 32],
                            [238, 136, 35],
                            [240, 72, 35]])
        palette = palette * 1.0 / 255
    elif imageID == 13:
        row = 1098
        col = 808
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 16:
        row = 1060
        col = 976
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [255, 255, 255]])
        palette = palette * 1.0 / 255
    elif imageID == 8:
        row = 601
        col = 1192
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80],
                            [127, 127, 127],
                            [85, 85, 85],
                            [26, 26, 26],
                            [46, 46, 46],
                            [210, 210, 210]])
        palette = palette * 1.0 / 255

    X_result = np.zeros((labels.shape[0], labels.shape[1], 3))
    for i in range(0, num_class + 1):
        # X_result[np.where(labels == i), 0] = palette[i, 0]
        # X_result[np.where(labels == i), 1] = palette[i, 1]
        # X_result[np.where(labels == i), 2] = palette[i, 2]
        X_result[np.where(labels == i)[0], np.where(labels == i)[1], 0] = palette[i, 0]
        X_result[np.where(labels == i)[0], np.where(labels == i)[1], 1] = palette[i, 1]
        X_result[np.where(labels == i)[0], np.where(labels == i)[1], 2] = palette[i, 2]

    X_result = np.reshape(X_result, (row, col, 3))

    return X_result


def test(net, img, patchsize, batchsize, Classes):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = patchsize
    center_pixel = True
    batch_size = batchsize
    n_classes = Classes
    pad_width = np.floor(patchsize / 2)
    pad_width = np.int(pad_width)
    kwargs = {'step': 1, 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    time = 0
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)
                data_point = data[:, :, :, pad_width, pad_width]

            indices = [b[1:] for b in batch]
            data = data.cuda()
            data_point = data_point.cuda()
            data_point = data_point.squeeze(1)
            data = data.squeeze(1)
            tim = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            # output= net(data)
            feature, output = net(data, data_point)
            end.record()
            torch.cuda.synchronize()
            tim = start.elapsed_time(end)
            time += tim
            # features = feature.view(feature.shape[0], -1)  # [b, -1]
            # distance = euclidean_dist(features, features)
            # distance = distance.to('cpu').numpy()
            # io.savemat('./distance.mat',{'distance':distance})
            if isinstance(output, tuple):
                output = output[1]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2, :] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs, time


probabilities, time = test(model, img, patchsize+1, batchsize, Classes)

time = time / 1000
print('GPU times is')
print(time)
prediction = np.argmax(probabilities, axis=-1)
prediction = prediction[pad_width:pad_width + m, pad_width:pad_width + n]
io.savemat('./GMA_net_result.mat', {'result': prediction})
io.loadmat('./GMA_net_result.mat')
# RGB = DrawResult(prediction, 16)
# RGB = DrawResult(prediction, 7)
# RGB = DrawResult(prediction, 3)
RGB = DrawResult(prediction, 5)
# plt.axis("off")
# plt.imshow(RGB)
# plt.savefig("./The_Img.png")
# # plt.figure(figsize=(571, 151))
# plt.show()

from PIL import Image
im = Image.fromarray(np.uint8(RGB*255))
im.save("CLMA-SA-20.jpg")
print("图片生成完毕")
