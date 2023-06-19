import numpy as np
import torch
import torch.utils
import torch.utils.data
class dataLoad(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, patch_size=25, center_pixel=True, flip_augmentation=True, radiation_augmentation=True, mixture_augmentation=True):
        """
        Args:
            data: 3D hyperspectral image---->(m, n, c)
            gt: 2D array of labels---->(m,n)
            patch_size: 图像块大小
            center_pixel: bool, 中心像素确定label
            flip_augmentation: bool, 随机左右、上下翻转
            radiation_augmentation:bool, 随机辐射增强
            mixture_augmentation：bool,
        """
        super(dataLoad, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        mask[gt == 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5   #范围（0-1）
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = np.fliplr(arrays)  #左右翻转
        if vertical:
            arrays = np.flipud(arrays)  #上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]  #原坐标
        x_center = x + self.patch_size // 2  # 边界扩充后的坐标
        y_center = y + self.patch_size // 2

        x1 = x_center - self.patch_size // 2  # 最上
        y1 = y_center - self.patch_size // 2  # 最左
        x2 = x_center + self.patch_size // 2  # 最下
        y2 = y_center + self.patch_size // 2  # 最右

        data = self.data[x1:x2+1, y1:y2+1]
        label = self.label[x, y]

        if self.flip_augmentation and self.patch_size > 1:
            data = self.flip(data)
        if self.radiation_augmentation:
            data = self.radiation_noise(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        data1 = data[:, self.patch_size // 2, self.patch_size // 2]  #
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        data1 = torch.from_numpy(data1)
        label = torch.from_numpy(label)

        label = label - 1
        return data, data1, label

class dataLoad1(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, patch_size=16, center_pixel=True, flip_augmentation=True, radiation_augmentation=True, mixture_augmentation=True):
        """
        Args:
            data: 3D hyperspectral image---->(m, n, c)
            gt: 2D array of labels---->(m,n)
            patch_size: 图像块大小
            center_pixel: bool, 中心像素确定label
            flip_augmentation: bool, 随机左右、上下翻转
            radiation_augmentation:bool, 随机辐射增强
            mixture_augmentation：bool,
        """
        super(dataLoad1, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        mask[gt == 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5   #范围（0-1）
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = np.fliplr(arrays)  #左右翻转
        if vertical:
            arrays = np.flipud(arrays)  #上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]  #原坐标
        x_center = x + self.patch_size // 2  #边界扩充后的坐标
        y_center = y + self.patch_size // 2
        x1, y1 = x_center - self.patch_size // 2, y_center - self.patch_size // 2  #左上角和右下角的坐标
        x2, y2 = x_center + self.patch_size // 2, y_center + self.patch_size // 2

        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.flip_augmentation and self.patch_size > 1:
            data = self.flip(data)
        if self.radiation_augmentation:
            data = self.radiation_noise(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        label = label - 1
        return data, label
class dataLoad2(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, patch_size=16, center_pixel=True, flip_augmentation=True, radiation_augmentation=True, mixture_augmentation=True):
        """
        Args:
            data: 3D hyperspectral image---->(m, n, c)
            gt: 2D array of labels---->(m,n)
            patch_size: 图像块大小
            center_pixel: bool, 中心像素确定label
            flip_augmentation: bool, 随机左右、上下翻转
            radiation_augmentation:bool, 随机辐射增强
            mixture_augmentation：bool,
        """
        super(dataLoad2, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        mask[gt == 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5   #范围（0-1）
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = np.fliplr(arrays)  #左右翻转
        if vertical:
            arrays = np.flipud(arrays)  #上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]  #原坐标
        x_center = x + self.patch_size // 2  #边界扩充后的坐标
        y_center = y + self.patch_size // 2
        x1, y1 = x_center - self.patch_size // 2, y_center - self.patch_size // 2  #左上角和右下角的坐标
        x2, y2 = x_center + self.patch_size // 2, y_center + self.patch_size // 2

        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.flip_augmentation and self.patch_size > 1:
            data = self.flip(data)
        if self.radiation_augmentation:
            data = self.radiation_noise(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        label = label - 1
        return data, label