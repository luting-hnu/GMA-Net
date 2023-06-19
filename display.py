import numpy as np
import torch
import itertools
from tqdm import tqdm
from scipy import io
# Visualization
import seaborn as sns
import visdom
import torch.nn as nn
import torch.nn.functional as F
from utils import convert_to_color_, convert_from_color_
from PIL import Image

patchsize= 31
class Res2(nn.Module):
    def __init__(self, in_channels, inter_channels, kernel_size, padding=0):
        super(Res2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        return X

class Res(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, groups=1):
        super(Res, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.res2 = Res2(in_channels, 32, kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Z = self.res2(X)
        return F.relu(X + Y + Z)


class sprn(nn.Module):
    def __init__(self, bands, classes, groups, groups_width, spa=False):
        super(sprn, self).__init__()
        self.bands = bands
        self.classes = classes
        self.spa = spa
        fc_planes = 128

        # pad the bands with final values
        new_bands = math.ceil(bands / groups) * groups
        pad_size = new_bands - bands
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, pad_size))

        # HPDM
        # if self.spa:
        #     self.spa_att = HPDM(new_bands)

            # SPRN
        self.conv1 = nn.Conv2d(new_bands, groups * groups_width, (1, 1), groups=groups)
        self.bn1 = nn.BatchNorm2d(groups * groups_width)

        self.res0 = Res(groups * groups_width, (1, 1), (0, 0), groups=groups)
        self.res1 = Res(groups * groups_width, (1, 1), (0, 0), groups=groups)

        self.conv2 = nn.Conv2d(groups_width * groups, fc_planes, (1, 1))
        self.bn2 = nn.BatchNorm2d(fc_planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_fc = nn.Linear(fc_planes, classes)

    def forward(self, x):
        # input: (b, 1, d, w, h)
        x = self.pad(x).squeeze(1)
        if self.spa:
            x = self.spa_att(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res0(x)
        x = self.res1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x).flatten(1)
        x = self.out_fc(x)
        return x


DATASET = "PaviaU"  # "KSC"
MODEL = "My_method"
Net_Name = './net_params_myNet_HS.pkl'

DataPath = './Data/PaviaU/PaviaU.mat'
GTPath = './Data/PaviaU/PaviaU_gt.mat'

Data = io.loadmat(DataPath)
GT = io.loadmat(GTPath)
# Data = Data['indian_pines_corrected']
Data = Data['paviaU']
# Data = Data['img']
# Data = Data['salinas_corrected']
# Data = Data['KSC']
Data = Data.astype(np.float32)
# GT = GT['KSC_gt']
GT = GT['paviaU_gt']
[m, n, l] = np.shape(Data)
# normalization
for i in range(l):
    Data[:, :, i] = (Data[:, :, i]-Data[:, :, i].min()) / (Data[:, :, i].max()-Data[:, :, i].min())
##load model
# model = torch.load(Net_Name)

# 数据边界填充，准备分割数据块
temp = Data[:, :, 0]
pad_width = np.floor(patchsize/2)
pad_width = np.int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2, n2] = temp2.shape
x2 = np.empty((m2, n2, l), dtype='float32')  # 待填充

for i in range(l):
    temp = Data[:, :, i]
    pad_width = np.floor(patchsize/2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x2[:, :, i] = temp2  # x2 为根据patch size四周填充后 patch size=30时，[610,340,103]->[640,370,103,]
Data = x2
# 数据边界填充完毕

from network import Network_bai, LliuMK_Net, LliuMK_Net2, LliuMK_Net3, LliuMK_Net4
model=LliuMK_Net3(classes=9, HSI_Data_Shape_H=610, HSI_Data_Shape_W=340, HSI_Data_Shape_C=103)
model.load_state_dict(torch.load(Net_Name))

img = Data
gt = GT

batchsize= 100
Classes = len(np.unique(gt))-1  # 10
# Generate color palette
palette = {0: (0, 0, 0)}
for k, color in enumerate(sns.color_palette("hls", Classes - 1)):
    palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)
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
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
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

def test(net, img, patchsize, batchsize,Classes):
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
                data_point = data[:,:,:,pad_width,pad_width]

            indices = [b[1:] for b in batch]
            # data = data.cuda()
            data = data.squeeze(1)
            output, _ = net(data,data)
            # output = net(data,data_point)
            if isinstance(output, tuple):
                output = output[0]
            # output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2,:] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs


probabilities = test(model, img, patchsize, batchsize, Classes)
prediction = np.argmax(probabilities, axis=-1)
# mask = np.zeros(gt.shape, dtype='bool')
# mask[gt == 0] = True
# prediction[mask] = 0
# color_prediction = convert_to_color(prediction)
# viz = visdom.Visdom(env=DATASET + ' ' + MODEL)
# if not viz.check_connection:
#     print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
# viz.images([np.transpose(color_prediction, (2, 0, 1))], opts={'caption': "Prediction"})


matrix3d = prediction
matrix3d = matrix3d+1

no_Row = matrix3d.shape[0]
no_column = matrix3d.shape[1]

aa = 0  # R
bb = 1  # G
cc = 2  # B

rgb_matrix = np.zeros(no_Row*no_column*3).reshape(no_Row, no_column, 3)
print(type(rgb_matrix), rgb_matrix.shape)
for i in range(0, no_Row):
    for j in range(0, no_column):
        if matrix3d[i][j] == 1:
            rgb_matrix[i][j][0] = 255;    rgb_matrix[i][j][1] = 0;   rgb_matrix[i][j][2] = 0
        elif matrix3d[i][j] == 2:
            rgb_matrix[i][j][0] = 0;    rgb_matrix[i][j][1] = 255;   rgb_matrix[i][j][2] = 0
        elif matrix3d[i][j] == 3:
            rgb_matrix[i][j][0] = 0;  rgb_matrix[i][j][1] = 0;     rgb_matrix[i][j][2] = 255
        elif matrix3d[i][j] == 4:
            rgb_matrix[i][j][0] = 255;    rgb_matrix[i][j][1] = 255;     rgb_matrix[i][j][2] = 0
        elif matrix3d[i][j] == 5:
            rgb_matrix[i][j][0] = 0;  rgb_matrix[i][j][1] = 255;     rgb_matrix[i][j][2] = 255
        elif matrix3d[i][j] == 6:
            rgb_matrix[i][j][0] = 255;  rgb_matrix[i][j][1] = 0;   rgb_matrix[i][j][2] = 255
        elif matrix3d[i][j] == 7:
            rgb_matrix[i][j][0] = 176;  rgb_matrix[i][j][1] = 48;   rgb_matrix[i][j][2] = 96
        elif matrix3d[i][j] == 8:
            rgb_matrix[i][j][0] = 46;  rgb_matrix[i][j][1] = 139;   rgb_matrix[i][j][2] = 87
        elif matrix3d[i][j] == 9:
            rgb_matrix[i][j][0] = 160;  rgb_matrix[i][j][1] = 32;   rgb_matrix[i][j][2] = 240

print(type(rgb_matrix))
print(rgb_matrix.shape)
print('*'*30)

"""r,g,b三个颜色通道可能不是一一对应的，如下进行修改调整"""
# rgb_matrix_image = np.zeros(no_Row*no_column*3).reshape(no_Row,no_column, 3)
# rgb_matrix_image[:, :, 0] = rgb_matrix[:, :, ]
# rgb_matrix_image[:, :, 1] = rgb_matrix[:, :, ]
# rgb_matrix_image[:, :, 2] = rgb_matrix[:, :, ]

print(pad_width) #15
print(pad_width+gt.shape[0], pad_width+gt.shape[1])  # 610,340
rgb_matrix = rgb_matrix[pad_width:pad_width+gt.shape[0]-1, pad_width:pad_width+gt.shape[1]-1]
rgb_matrix_image = Image.fromarray(np.uint8(rgb_matrix))  # 将之前的矩阵转换为图片
rgb_matrix_image.show()  # 调用本地软件显示图片，win10是叫照片的工具
# rgb_matrix_image.save('result_model_cls_ALL_withoutAttention.tif')
rgb_matrix_image.save('Proposed4.tif')
