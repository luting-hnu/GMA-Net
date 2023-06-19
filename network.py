"""该模块中用于保存网络模型"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import spectral
import numpy
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.decomposition import PCA



"""----------------------------------------我的网络3-------------------------------------------------"""


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


class CLMA_Net(nn.Module):
    def __init__(self, classes, HSI_Data_Shape_H, HSI_Data_Shape_W, HSI_Data_Shape_C):  # band:103  classes=9
        super(CLMA_Net, self).__init__()
        self.name = 'LliuMK_Net3'
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C

        # self.mish = mish()  # 也可以引用一下，等待后续改进
        self.relu = nn.ReLU()

        self.CA1 = ChannelAttention_LliuMK3(in_planes=self.band)

        self.MCA1 = MultiChannelAttention_LliuMK3(in_planes=self.band)
        self.MCA2 = MultiChannelAttention_LliuMK3(in_planes=self.band)

        self.SA1 = SpatialAttention_LliuMK3()

        self.MSA1 = MultiSpatialAttention_LliuMK3()
        self.MSA2 = MultiSpatialAttention_LliuMK3()
        self.conv11 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm11 = nn.BatchNorm2d(self.band)

        self.conv12 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm12 = nn.BatchNorm2d(self.band)

        self.conv13 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm13 = nn.BatchNorm2d(self.band)

        self.conv21 = nn.Conv2d(in_channels=self.band, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
        self.batch_norm21 = nn.BatchNorm2d(64)

        self.conv22 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch_norm22 = nn.BatchNorm2d(128)

        self.conv23 = nn.Conv2d(in_channels=128, out_channels=self.band, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
        self.batch_norm23 = nn.BatchNorm2d(self.band)


        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.finally_fc_classification = nn.Linear(self.band * 2, self.classes)

    def forward(self, patchX, pixelX):  # x:(16,103,9,9)
        """------------------------光谱分支------------------------"""
        patch_size = patchX.shape[-1] // 2
        input_spectral = patchX[:, :, (patch_size-1):(patch_size + 2), (patch_size-1):(patch_size + 2)]  # [64,103,3,3]

        # input_spectral = input_spectral.permute(0, 2, 1, 3)  # [64,3,103,3]
        x11 = self.conv11(input_spectral)
        x11 = self.batch_norm11(x11)  # [64,103,3,3]
        x11 = self.relu(x11)
        ca1 = self.CA1(x11)
        x11 = x11 * ca1

        x12 = x11
        x12 = self.conv12(x12)
        x12 = self.batch_norm12(x12)
        x12 = self.relu(x12)
        mca1 = self.MCA1(x12, ca1)
        x12 = x12 * mca1

        x13 = x12
        x13 = self.conv13(x13)
        x13 = self.batch_norm13(x13)
        x13 = self.relu(x13)
        mca2 = self.MCA2(x13, mca1)
        x13 = x13 * mca2

        x13 = self.global_pooling(x13)
        x13 = x13.view(x13.size(0), -1)
        output_spectral = x13

        """------------------------空间分支------------------------"""
        input_spatial = patchX
        x21 = self.conv21(input_spatial)  # (16,32,9,9)<—(16,103,9,9)
        x21 = self.batch_norm21(x21)  # (16,32,9,9)
        x21 = self.relu(x21)  # (16,32,9,9)
        sa1 = self.SA1(x21)
        x21 = x21 * sa1
        x21 = nn.MaxPool2d(2)(x21)

        x22 = self.conv22(x21)  # (16,24,1,9,9)
        x22 = self.batch_norm22(x22)  # (16,24,1,9,9)
        x22 = self.relu(x22)
        msa1 = self.MSA1(x22, sa1)
        x22 = x22 * msa1

        x22 = nn.MaxPool2d(2)(x22)

        x23 = self.conv23(x22)  # (16,24,1,9,9)
        x23 = self.batch_norm23(x23)  # (16,24,1,9,9)
        x23 = self.relu(x23)
        msa2 = self.MSA2(x23, msa1)
        x23 = x23 * msa2

        x23 = nn.MaxPool2d(2)(x23)

        x25 = self.global_pooling(x23)
        x25 = x25.view(x25.size(0), -1)
        output_spatial = x25

        output = torch.cat((output_spectral, output_spatial), dim=1)
        output = self.finally_fc_classification(output)
        output = F.softmax(output, dim=1)

        return output, output


"""-------------------------------GMA-----------------------------------------"""

"""↓↓↓光谱注意力↓↓↓"""

class Spectral_Attention_block(nn.Module):
    def __init__(self, dim=26):
        super(Spectral_Attention_block, self).__init__()
        self.dim = dim

        self.key_embed = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1),
                      padding=(0, 0), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU()
        )

        self.value_embed = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1), bias=False),  # 1*1的卷积进行Value的编码
            nn.BatchNorm2d(self.dim)
        )

        self.factor = 2
        self.attention_embed = nn.Sequential(  # 通过连续两个1*1的卷积计算注意力矩阵
            nn.Conv2d(2 * self.dim, self.dim // 4, (1, 1), bias=False),
            nn.BatchNorm2d(self.dim // 4),
            nn.ReLU(),
            nn.Conv2d(self.dim // 4, self.factor * self.factor * self.dim, (1, 1), stride=(1, 1))
        )

        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        bs, c, h, w = x.shape  # 64,26,15,15
        k1 = self.key_embed(x)  # shape：bs,c,h,w  提取静态上下文信息得到key

        v = self.value_embed(x).view(bs, c, -1)  # shape：bs,c,h*w  得到value编码

        y = torch.cat([k1, x], dim=1)  # shape：bs,2c,h,w  Key与Query在channel维度上进行拼接进行拼接
        att = self.attention_embed(y)  # shape：bs,c*k*k,h,w  计算注意力矩阵
        att = att.reshape(bs, c, self.factor * self.factor, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # shape：bs,c,h*w  求平均降低维度
        k2 = F.softmax(att, dim=-1) * v  # 对每一个H*w进行softmax后
        k2 = k2.view(bs, c, h, w)

        out = self.coefficient1 * x + self.coefficient2 * k2
        return out  # 注意力融合


"""↓↓↓空间注意力↓↓↓"""


class MultiGroupSpatialAttention(nn.Module):
    r"""空间注意力"""

    def __init__(self, channel, patch_size):
        super(MultiGroupSpatialAttention, self).__init__()
        self.channel = channel
        self.patch_size = patch_size

        self.attn_conv = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.up_sample = nn.Upsample(mode='nearest', size=(self.channel, self.patch_size, self.patch_size))
        self.conv = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.bn = nn.BatchNorm2d(num_features=self.channel)

    def forward(self, input_x):
        x = input_x
        x = x.unsqueeze(1)
        x = self.attn_conv(x)
        x = self.up_sample(x)

        x = self.conv(x)
        x = x.squeeze(1)
        x = self.bn(x)
        x = self.sigmoid(x)
        output = input_x * x + input_x
        return output, x


class Multi_Attention(nn.Module):
    def __init__(self, channelNumber, patch_size=31):
        super(Multi_Attention, self).__init__()
        self.total_Channel_num = channelNumber
        self.patch_size = patch_size
        """===========================PaviaU==========================="""
        self.mgca01 = Spectral_Attention_block(dim=73)
        self.mgca02 = Spectral_Attention_block(dim=11)
        self.mgca03 = Spectral_Attention_block(dim=19)

        self.mgsa01 = MultiGroupSpatialAttention(channel=73, patch_size=self.patch_size)
        self.mgsa02 = MultiGroupSpatialAttention(channel=11, patch_size=self.patch_size)
        self.mgsa03 = MultiGroupSpatialAttention(channel=19, patch_size=self.patch_size)
        """===========================Houston==========================="""
        # self.mgca01 = Spectral_Attention_block(dim=85)
        # self.mgca02 = Spectral_Attention_block(dim=35)
        # self.mgca03 = Spectral_Attention_block(dim=24)
        #
        # self.mgsa01 = MultiGroupSpatialAttention(channel=85, patch_size=self.patch_size)
        # self.mgsa02 = MultiGroupSpatialAttention(channel=35, patch_size=self.patch_size)
        # self.mgsa03 = MultiGroupSpatialAttention(channel=24, patch_size=self.patch_size)
        """===========================Salinas==========================="""
        # self.mgca01 = Spectral_Attention_block(dim=38)
        # self.mgca02 = Spectral_Attention_block(dim=45)
        # self.mgca03 = Spectral_Attention_block(dim=23)
        # self.mgca04 = Spectral_Attention_block(dim=42)
        # self.mgca05 = Spectral_Attention_block(dim=56)
        #
        # self.mgsa01 = MultiGroupSpatialAttention(channel=38, patch_size=self.patch_size)
        # self.mgsa02 = MultiGroupSpatialAttention(channel=45, patch_size=self.patch_size)
        # self.mgsa03 = MultiGroupSpatialAttention(channel=23, patch_size=self.patch_size)
        # self.mgsa04 = MultiGroupSpatialAttention(channel=42, patch_size=self.patch_size)
        # self.mgsa05 = MultiGroupSpatialAttention(channel=56, patch_size=self.patch_size)
        """===========================Indian_pines==========================="""
        # self.mgca01 = Spectral_Attention_block(dim=35)
        # self.mgca02 = Spectral_Attention_block(dim=21)
        # self.mgca03 = Spectral_Attention_block(dim=26)
        # self.mgca04 = Spectral_Attention_block(dim=21)
        # self.mgca05 = Spectral_Attention_block(dim=41)
        # self.mgca06 = Spectral_Attention_block(dim=56)
        #
        # self.mgsa01 = MultiGroupSpatialAttention(channel=35, patch_size=self.patch_size)
        # self.mgsa02 = MultiGroupSpatialAttention(channel=21, patch_size=self.patch_size)
        # self.mgsa03 = MultiGroupSpatialAttention(channel=26, patch_size=self.patch_size)
        # self.mgsa04 = MultiGroupSpatialAttention(channel=21, patch_size=self.patch_size)
        # self.mgsa05 = MultiGroupSpatialAttention(channel=41, patch_size=self.patch_size)
        # self.mgsa06 = MultiGroupSpatialAttention(channel=56, patch_size=self.patch_size)
        """===========================KSC==========================="""
        # self.mgca01 = Spectral_Attention_block(dim=35)
        # self.mgca02 = Spectral_Attention_block(dim=21)
        # self.mgca03 = Spectral_Attention_block(dim=26)
        # self.mgca04 = Spectral_Attention_block(dim=21)
        # self.mgca05 = Spectral_Attention_block(dim=41)
        # self.mgca06 = Spectral_Attention_block(dim=56)
        #
        # self.mgsa01 = MultiGroupSpatialAttention(channel=35, patch_size=self.patch_size)
        # self.mgsa02 = MultiGroupSpatialAttention(channel=21, patch_size=self.patch_size)
        # self.mgsa03 = MultiGroupSpatialAttention(channel=26, patch_size=self.patch_size)
        # self.mgsa04 = MultiGroupSpatialAttention(channel=21, patch_size=self.patch_size)
        # self.mgsa05 = MultiGroupSpatialAttention(channel=41, patch_size=self.patch_size)
        # self.mgsa06 = MultiGroupSpatialAttention(channel=56, patch_size=self.patch_size)
        """========================================================================="""

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.total_Channel_num * 2, out_channels=self.total_Channel_num, kernel_size=(1, 1),
                      padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(self.total_Channel_num),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.total_Channel_num, out_channels=32, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        """注意改系数，PaviaU:3, Houston:3, Salinas:5, IndianPines:6"""
        self.safusion_conv_layer = nn.Sequential(
            nn.Conv2d(3 * self.total_Channel_num, self.total_Channel_num, (1, 1)),
            nn.BatchNorm2d(self.total_Channel_num),
            nn.ReLU()
        )

    def forward(self, x):
        """===========================paviau==========================="""
        """---------------------------Group01--------------------------"""
        group01_ca_output = self.mgca01(x[:, 0:73, :, :])
        group01_sa_output, sa1 = self.mgsa01(x[:, 0:73, :, :])
        sap1 = torch.mean(sa1, dim=1).unsqueeze(1)
        """---------------------------Group02--------------------------"""
        group02_ca_output = self.mgca02(x[:, 73:84, :, :])
        group02_sa_output, sa2 = self.mgsa02(x[:, 73:84, :, :])
        sap2 = torch.mean(sa2, dim=1).unsqueeze(1)
        """---------------------------Group03--------------------------"""
        group03_ca_output = self.mgca03(x[:, 84:103, :, :])
        group03_sa_output, sa3 = self.mgsa03(x[:, 84:103, :, :])
        sap3 = torch.mean(sa3, dim=1).unsqueeze(1)

        ca_output = torch.cat([group01_ca_output, group02_ca_output, group03_ca_output], dim=1)

        safusion1 = torch.cat(
            [sa1 * (x[:, 0:73, :, :]), (1 - sap1) * (x[:, 73:84, :, :]), (1 - sap1) * (x[:, 84:103, :, :])], dim=1)
        safusion2 = torch.cat(
            [(1 - sap2) * (x[:, 0:73, :, :]), sa2 * (x[:, 73:84, :, :]), (1 - sap2) * (x[:, 84:103, :, :])], dim=1)
        safusion3 = torch.cat(
            [(1 - sap3) * (x[:, 0:73, :, :]), (1 - sap3) * (x[:, 73:84, :, :]), sa3 * (x[:, 84:103, :, :])], dim=1)
        safusion_conv = self.safusion_conv_layer(torch.cat([safusion1, safusion2, safusion3], dim=1))
        safusion_conv = safusion_conv + x

        """===========================Houston=========================="""
        # """---------------------------Group01--------------------------"""
        # group01_ca_output = self.mgca01(x[:, 0:85, :, :])
        # group01_sa_output, sa1 = self.mgsa01(x[:, 0:85, :, :])
        # sap1 = torch.mean(sa1, dim=1).unsqueeze(1)
        # """---------------------------Group02--------------------------"""
        # group02_ca_output = self.mgca02(x[:, 85:120, :, :])
        # group02_sa_output, sa2 = self.mgsa02(x[:, 85:120, :, :])
        # sap2 = torch.mean(sa2, dim=1).unsqueeze(1)
        # """---------------------------Group03--------------------------"""
        # group03_ca_output = self.mgca03(x[:, 120:144, :, :])
        # group03_sa_output, sa3 = self.mgsa03(x[:, 120:144, :, :])
        # sap3 = torch.mean(sa3, dim=1).unsqueeze(1)
        #
        # safusion1 = torch.cat(
        #     [sa1 * (x[:, 0:85, :, :]),
        #      (1 - sap1) * (x[:, 85:120, :, :]),
        #      (1 - sap1) * (x[:, 120:144, :, :])], dim=1)
        # safusion2 = torch.cat(
        #     [(1 - sap2) * (x[:, 0:85, :, :]),
        #      sa2 * (x[:, 85:120, :, :]),
        #      (1 - sap2) * (x[:, 120:144, :, :])], dim=1)
        # safusion3 = torch.cat(
        #     [(1 - sap3) * (x[:, 0:85, :, :]),
        #      (1 - sap3) * (x[:, 85:120, :, :]),
        #      sa3 * (x[:, 120:144, :, :])], dim=1)
        # safusion_conv = self.safusion_conv_layer(torch.cat([safusion1, safusion2, safusion3], dim=1))
        # safusion_conv = safusion_conv + x
        # ca_output = torch.cat([group01_ca_output, group02_ca_output, group03_ca_output], dim=1)
        # # sa_output = torch.cat([group01_sa_output, group02_sa_output, group03_sa_output], dim=1)
        """===========================Salinas==========================="""
        # """---------------------------Group01--------------------------"""
        # group01_ca_output = self.mgca01(x[:, 0:38, :, :])
        # group01_sa_output, sa1 = self.mgsa01(x[:, 0:38, :, :])
        # sap1 = torch.mean(sa1, dim=1).unsqueeze(1)
        # """---------------------------Group02--------------------------"""
        # group02_ca_output = self.mgca02(x[:, 38:83, :, :])
        # group02_sa_output, sa2 = self.mgsa02(x[:, 38:83, :, :])
        # sap2 = torch.mean(sa2, dim=1).unsqueeze(1)
        # """---------------------------Group03--------------------------"""
        # group03_ca_output = self.mgca03(x[:, 83:106, :, :])
        # group03_sa_output, sa3 = self.mgsa03(x[:, 83:106, :, :])
        # sap3 = torch.mean(sa3, dim=1).unsqueeze(1)
        # """---------------------------Group04--------------------------"""
        # group04_ca_output = self.mgca04(x[:, 106:148, :, :])
        # group04_sa_output, sa4 = self.mgsa04(x[:, 106:148, :, :])
        # sap4 = torch.mean(sa4, dim=1).unsqueeze(1)
        # """---------------------------Group05--------------------------"""
        # group05_ca_output = self.mgca05(x[:, 148:204, :, :])
        # group05_sa_output, sa5 = self.mgsa05(x[:, 148:204, :, :])
        # sap5 = torch.mean(sa5, dim=1).unsqueeze(1)
        #
        # # ca_output = torch.cat([group01_ca_output, group02_ca_output, group03_ca_output], dim=1)
        # ca_output = torch.cat(
        #     [group01_ca_output, group02_ca_output, group03_ca_output, group04_ca_output, group05_ca_output], dim=1)
        #
        # safusion1 = torch.cat(
        #     [sa1 * (x[:, 0:38, :, :]),
        #      (1 - sap1) * (x[:, 38:83, :, :]),
        #      (1 - sap1) * (x[:, 83:106, :, :]),
        #      (1 - sap1) * (x[:, 106:148, :, :]),
        #      (1 - sap1) * (x[:, 148:204, :, :])
        #      ], dim=1)
        # safusion2 = torch.cat(
        #     [(1 - sap2) * (x[:, 0:38, :, :]),
        #      sa2 * (x[:, 38:83, :, :]),
        #      (1 - sap2) * (x[:, 83:106, :, :]),
        #      (1 - sap2) * (x[:, 106:148, :, :]),
        #      (1 - sap2) * (x[:, 148:204, :, :])
        #      ], dim=1)
        # safusion3 = torch.cat(
        #     [(1 - sap3) * (x[:, 0:38, :, :]),
        #      (1 - sap3) * (x[:, 38:83, :, :]),
        #      sa3 * (x[:, 83:106, :, :]),
        #      (1 - sap3) * (x[:, 106:148, :, :]),
        #      (1 - sap3) * (x[:, 148:204, :, :])
        #      ], dim=1)
        # safusion4 = torch.cat(
        #     [(1 - sap4) * (x[:, 0:38, :, :]),
        #      (1 - sap4) * (x[:, 38:83, :, :]),
        #      (1 - sap4) * (x[:, 83:106, :, :]),
        #      sa4 * (x[:, 106:148, :, :]),
        #      (1 - sap4) * (x[:, 148:204, :, :])
        #      ], dim=1)
        # safusion5 = torch.cat(
        #     [(1 - sap5) * (x[:, 0:38, :, :]),
        #      (1 - sap5) * (x[:, 38:83, :, :]),
        #      (1 - sap5) * (x[:, 83:106, :, :]),
        #      (1 - sap5) * (x[:, 106:148, :, :]),
        #      sa5 * (x[:, 148:204, :, :])
        #      ], dim=1)
        #
        # safusion_conv = self.safusion_conv_layer(
        #     torch.cat([safusion1, safusion2, safusion3, safusion4, safusion5], dim=1))
        #
        # safusion_conv = safusion_conv + x
        """===========================Indian_pines==========================="""
        # """---------------------------Group01--------------------------"""
        # group01_ca_output = self.mgca01(x[:, 0:35, :, :])
        # group01_sa_output, sa1 = self.mgsa01(x[:, 0:35, :, :])
        # sap1 = torch.mean(sa1, dim=1).unsqueeze(1)
        # """---------------------------Group02--------------------------"""
        # group02_ca_output = self.mgca02(x[:, 35:56, :, :])
        # group02_sa_output, sa2 = self.mgsa02(x[:, 35:56, :, :])
        # sap2 = torch.mean(sa2, dim=1).unsqueeze(1)
        # """---------------------------Group03--------------------------"""
        # group03_ca_output = self.mgca03(x[:, 56:82, :, :])
        # group03_sa_output, sa3 = self.mgsa03(x[:, 56:82, :, :])
        # sap3 = torch.mean(sa3, dim=1).unsqueeze(1)
        # """---------------------------Group04--------------------------"""
        # group04_ca_output = self.mgca04(x[:, 82:103, :, :])
        # group04_sa_output, sa4 = self.mgsa04(x[:, 82:103, :, :])
        # sap4 = torch.mean(sa4, dim=1).unsqueeze(1)
        # """---------------------------Group05--------------------------"""
        # group05_ca_output = self.mgca05(x[:, 103:144, :, :])
        # group05_sa_output, sa5 = self.mgsa05(x[:, 103:144, :, :])
        # sap5 = torch.mean(sa5, dim=1).unsqueeze(1)
        # """---------------------------Group06--------------------------"""
        # group06_ca_output = self.mgca06(x[:, 144:200, :, :])
        # group06_sa_output, sa6 = self.mgsa06(x[:, 144:200, :, :])
        # sap6 = torch.mean(sa6, dim=1).unsqueeze(1)
        #
        # ca_output = torch.cat(
        #     [group01_ca_output, group02_ca_output, group03_ca_output, group04_ca_output, group05_ca_output,
        #      group06_ca_output], dim=1)
        # # sa_output = torch.cat(
        # #     [group01_sa_output, group02_sa_output, group03_sa_output, group04_sa_output, group05_sa_output,
        # #      group06_sa_output], dim=1)
        #
        # safusion1 = torch.cat(
        #     [sa1 * (x[:, 0:35, :, :]),
        #      (1 - sap1) * (x[:, 35:56, :, :]),
        #      (1 - sap1) * (x[:, 56:82, :, :]),
        #      (1 - sap1) * (x[:, 82:103, :, :]),
        #      (1 - sap1) * (x[:, 103:144, :, :]),
        #      (1 - sap1) * (x[:, 144:200, :, :])], dim=1)
        # safusion2 = torch.cat(
        #     [(1 - sap2) * (x[:, 0:35, :, :]),
        #      sa2 * (x[:, 35:56, :, :]),
        #      (1 - sap2) * (x[:, 56:82, :, :]),
        #      (1 - sap2) * (x[:, 82:103, :, :]),
        #      (1 - sap2) * (x[:, 103:144, :, :]),
        #      (1 - sap2) * (x[:, 144:200, :, :])], dim=1)
        # safusion3 = torch.cat(
        #     [(1 - sap3) * (x[:, 0:35, :, :]),
        #      (1 - sap3) * (x[:, 35:56, :, :]),
        #      sa3 * (x[:, 56:82, :, :]),
        #      (1 - sap3) * (x[:, 82:103, :, :]),
        #      (1 - sap3) * (x[:, 103:144, :, :]),
        #      (1 - sap3) * (x[:, 144:200, :, :])], dim=1)
        # safusion4 = torch.cat(
        #     [(1 - sap4) * (x[:, 0:35, :, :]),
        #      (1 - sap4) * (x[:, 35:56, :, :]),
        #      (1 - sap4) * (x[:, 56:82, :, :]),
        #      sa4 * (x[:, 82:103, :, :]),
        #      (1 - sap4) * (x[:, 103:144, :, :]),
        #      (1 - sap4) * (x[:, 144:200, :, :])], dim=1)
        # safusion5 = torch.cat(
        #     [(1 - sap5) * (x[:, 0:35, :, :]),
        #      (1 - sap5) * (x[:, 35:56, :, :]),
        #      (1 - sap5) * (x[:, 56:82, :, :]),
        #      (1 - sap5) * (x[:, 82:103, :, :]),
        #      sa5 * (x[:, 103:144, :, :]),
        #      (1 - sap5) * (x[:, 144:200, :, :])], dim=1)
        # safusion6 = torch.cat(
        #     [(1 - sap6) * (x[:, 0:35, :, :]),
        #      (1 - sap6) * (x[:, 35:56, :, :]),
        #      (1 - sap6) * (x[:, 56:82, :, :]),
        #      (1 - sap6) * (x[:, 82:103, :, :]),
        #      (1 - sap6) * (x[:, 103:144, :, :]),
        #      sa6 * (x[:, 144:200, :, :])], dim=1)
        # safusion_conv = self.safusion_conv_layer(torch.cat([safusion1, safusion2, safusion3, safusion4, safusion5, safusion6], dim=1))
        # safusion_conv = safusion_conv + x

        """===========================KSC==========================="""
        # """---------------------------Group01--------------------------"""
        # group01_ca_output = self.mgca01(x[:, 0:35, :, :])
        # group01_sa_output = self.mgsa01(x[:, 0:35, :, :])
        # """---------------------------Group02--------------------------"""
        # group02_ca_output = self.mgca02(x[:, 35:56, :, :])
        # group02_sa_output = self.mgsa02(x[:, 35:56, :, :])
        # """---------------------------Group03--------------------------"""
        # group03_ca_output = self.mgca03(x[:, 56:82, :, :])
        # group03_sa_output = self.mgsa03(x[:, 56:82, :, :])
        # """---------------------------Group04--------------------------"""
        # group04_ca_output = self.mgca04(x[:, 82:103, :, :])
        # group04_sa_output = self.mgsa04(x[:, 82:103, :, :])
        # """---------------------------Group05--------------------------"""
        # group05_ca_output = self.mgca05(x[:, 103:144, :, :])
        # group05_sa_output = self.mgsa05(x[:, 103:144, :, :])
        # """---------------------------Group06--------------------------"""
        # group06_ca_output = self.mgca06(x[:, 144:200, :, :])
        # group06_sa_output = self.mgsa06(x[:, 144:200, :, :])
        #
        # ca_output = torch.cat(
        #     [group01_ca_output, group02_ca_output, group03_ca_output, group04_ca_output, group05_ca_output,
        #      group06_ca_output], dim=1)
        # sa_output = torch.cat(
        #     [group01_sa_output, group02_sa_output, group03_sa_output, group04_sa_output, group05_sa_output,
        #      group06_sa_output], dim=1)
        """=================================================================="""

        output = torch.cat([safusion_conv, ca_output], dim=1)
        output = self.fusion_layer(output)

        return output


class GMA_Net(nn.Module):
    def __init__(self, classes, HSI_Data_Shape_H, HSI_Data_Shape_W, HSI_Data_Shape_C,
                 patch_size):  # band:103  classes=9
        super(GMA_Net, self).__init__()
        self.name = 'GMA_Net'
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C
        self.patch_size = patch_size

        self.gma = Multi_Attention(channelNumber=self.band, patch_size=self.patch_size)

        # self.mish = mish()  # 也可以引用一下，等待后续改进
        self.relu = nn.ReLU()

        """Pixel branch"""
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=self.band, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        """Patch branch"""
        self.conv31 = nn.Conv2d(in_channels=self.band, out_channels=32, kernel_size=(5, 5), stride=(1, 1),
                                padding=(2, 2))
        self.batch_norm31 = nn.BatchNorm2d(32)

        self.conv32 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.batch_norm32 = nn.BatchNorm2d(64)

        self.conv33 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1),
                                padding=(2, 2))
        self.batch_norm33 = nn.BatchNorm2d(128)

        self.conv34 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1),
                                padding=(2, 2))
        self.batch_norm34 = nn.BatchNorm2d(256)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.finally_fc_classification = nn.Linear(256 * 2, self.classes)

    def forward(self, patchX, pixelX):  # x:(64,103,31,31)
        """------------------------branch 1 (1×1)------------------------"""
        input_1 = pixelX.unsqueeze(-1).unsqueeze(-1)
        x11 = self.conv11(input_1)

        x12 = self.conv12(x11)

        x13 = self.conv13(x12)

        x14 = self.conv14(x13)

        output_1 = x14.view(x14.size(0), -1)

        """------------------------branch 3 (31×31)------------------------"""
        input_3 = patchX
        input_3 = self.gma(input_3)

        x32 = nn.MaxPool2d(2)(input_3)
        x32 = self.conv32(x32)
        x32 = self.batch_norm32(x32)
        x32 = self.relu(x32)

        x33 = nn.MaxPool2d(2)(x32)
        x33 = self.conv33(x33)
        x33 = self.batch_norm33(x33)
        x33 = self.relu(x33)

        x34 = nn.MaxPool2d(2)(x33)
        x34 = self.conv34(x34)
        x34 = self.batch_norm34(x34)
        x34 = self.relu(x34)

        x34 = self.global_pooling(x34)
        output_3 = x34.view(x34.size(0), -1)
        """------------------------fusion------------------------"""
        output = torch.cat((output_1, output_3), dim=1)
        output = self.finally_fc_classification(output)
        output = F.softmax(output, dim=1)

        return output, output


"""----------------------------------------我的网络结束----------------------------------------"""

