import numpy as np
from PIL import Image
import scipy.io as io
import json

# numbers = [[2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
#            [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
#            [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
#            [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
#            [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
#            [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
#            ]

# file_path = './Predicted_result.json'
# with open(file_path) as file_object:
#     numbers = json.load(file_object)
#matrix3d = np.array(numbers)

# matrix3d = io.loadmat('./result_model_cls_ALL_withoutAttention.mat')
# matrix3d = io.loadmat('./prediction.mat')['data']
matrix3d = io.loadmat('./Data/PaviaU/PaviaU_gt.mat')['paviaU_gt']
# matrix3d = prediction
# matrix3d = matrix3d+1

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

# print(pad_width) #15
# print(pad_width+gt.shape[0], pad_width+gt.shape[1])  # 610,340
# rgb_matrix = rgb_matrix[pad_width:pad_width+gt.shape[0], pad_width:pad_width+gt.shape[1]]
rgb_matrix_image = Image.fromarray(np.uint8(rgb_matrix))  # 将之前的矩阵转换为图片
rgb_matrix_image.show()  # 调用本地软件显示图片，win10是叫照片的工具
# rgb_matrix_image.save('result_model_cls_ALL_withoutAttention.tif')
rgb_matrix_image.save('gt.tif')