# GMA-Net
This repository provides the code for the methods and experiments presented in our paper '**Grouped Multi-Attention Network for Hyperspectral
Image Spectral-Spatial Classification**'.
You can find the PDF of this paper on: https://ieeexplore.ieee.org/document/10091142

![net](https://github.com/luting-hnu/GMA-Net/blob/main/figure/GMA-Net.png)

**If you have any questions, you can send me an email. My mail address is liumengkai@hnu.edu.cn.**



## Directory structure

```
path to dataset:
                ├─Data
                  ├─PaviaU
                  	├─PaviaU.mat
                  	├─PaviaU_gt.mat
                  	├─PaviaU_eachClass_20_train_1.mat
                  	├─PaviaU_eachClass_20_test_1.mat
                  	...
                  ├─salinas
                  	├─Salinas_corrected.mat
                  	├─Salinas_gt.mat
                  	├─salinas_eachClass_20_train_1.mat
                  	├─salinas_eachClass_20_test_1.mat
                  	...
                  ├─Houston
                  	├─Houston.mat
                  	├─Houston_gt.mat
                  	├─Houston_eachClass_20_train_1.mat
                    ├─Houston_eachClass_20_test_1.mat
                    ...
```

## Generate experimental samples

```
split_Dateset.py
```

## Train

```
myNet_new.py
```

## Citation

If you find this paper useful, please cite:

```
Ting Lu, Mengkai Liu, Wei Fu and Xudong Kang, "Grouped Multi-Attention Network for Hyperspectral Image Spectral-Spatial Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-12, 2023, Art no. 5507912, doi: 10.1109/TGRS.2023.3263851.
```

