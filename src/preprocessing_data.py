import re
from abc import ABC
from os.path import join
from typing import List, Dict
import cv2
import monai
import numpy as np
import os

import torch.utils.data
from torch.utils.data import DataLoader
import glob
import tqdm as tqdm
import matplotlib.pyplot as plt

'''
AddChanneld：用于添加图像尺寸中的通道数（并不是增加通道数）
LoadNifti：用于加载Nifti格式的文件
LoadNiftid：也是用于加载Nifti格式的文件，但是是以字典的形式进行包装保存 调用
Orientationd：重定向轴方向
Rand3DElasticd：随机3D弹性变形
RandAffined：随机仿射变换
Spacingd：图像重采样
'''
from monai import transforms
from monai.data import Dataset, DataLoader, list_data_collate
from monai.transforms import (
    transform,
    EnsureChannelFirst,
    # AddChanneld,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Resized,
    Spacingd, LoadImaged, Compose, EnsureChannelFirstd, ToMetaTensord, NormalizeIntensityd, ToTensord, ScaleIntensityd,
    RandCropByPosNegLabeld, RandRotate90d,
)
import SimpleITK as sitk


# *****************************************************  compose_transform
# *******************************************************************

def get_data_base_transform():
    base_transform = [
        # [B, H, W, D] --> [B, C, H, W, D]
        transforms.AddChanneld(keys=['image', 'label']),
        transforms.Orientationd(keys=['image', 'label'], axcodes="RAS"),
        transforms.ToTensord(keys=['image', 'label']),
        # transforms.ConcatItemsd(keys=['image', 'label'],name='index', dim=0),
        # transforms.DeleteItemsd(keys=['image', 'label']),
        # RobustZScoreNormalization(keys=['flair', 't1', 't1ce', 't2']),

    ]
    return transforms.Compose(base_transform)


def get_data_train_transform():
    return transforms.Compose([
        transforms.EnsureChannelFirstd(keys=['image', 'label']),
        transforms.Orientationd(keys=['image', 'label'], axcodes="RAS"),
        # transforms.ConcatItemsd(keys=['image', 'label'],name='index' ,dim=0),
        # transforms.DeleteItemsd(keys=['image', 'label']),

        # crop
        # transforms.RandCropByPosNegLabeld(
        #     keys=["image", 'label'],
        #     label_key='label',
        #     spatial_size=3,
        #     pos=1.0,
        #     neg=1.0,
        #     num_samples=1),
        transforms.CropForegroundd(keys=["image", 'label'], source_key='image'),

        # spatial aug
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),

        # intensity aug
        transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),
        # norm
        NormalizeIntensityd(keys=["image", 'label'], nonzero=False, dtype=np.float32),

        transforms.Resized(keys=['image', 'label'], spatial_size=(320, 320, 32)),
        transforms.ToTensord(keys=['image', 'label']),

    ])


# *****************************************************  my_data_set_class
# *******************************************************************

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, model, data_transforms: object = None):
        super(MyDataset, self).__init__()
        assert model in ['train', 'test'], 'Unknown model'
        self.model = model
        self.root = root
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        self.traindata_path = []
        self.testdata_path = []

        with open(self.root, 'r') as f:
            # 148个人
            for one_data_path_dict in tqdm.tqdm(f.readlines()):
                if self.model == 'train':
                    self.traindata_path.append(one_data_path_dict)
                    for i in self.traindata_path:
                        if os.path.exists(i) is False:
                            raise FileNotFoundError(f"file {i} does not exists.")
                else:
                    self.testdata_path.append(one_data_path_dict)
                    for i in self.testdata_path:
                        if os.path.exists(i) is False:
                            raise FileNotFoundError(f"file {i} does not exists.")

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# *****************************************************  pre_data_loader
# *******************************************************************
train_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]),
        RandCropByPosNegLabeld(
            keys=["img", "seg"],
            label_key="seg",
            spatial_size=[96, 96, 96],
            pos=1,
            neg=1,
            num_samples=4,
        ),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 2)),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]),
    ]
)


def pre_test_dataset():
    data_path_dict = []
    with open('./src/valdata_path_dict.txt', 'r') as f:
        # 20个人
        for one_data_path_dict in tqdm.tqdm(f.readlines()):
            # print(one_data_path_dict)
            data_path_dict.append(one_data_path_dict)
    # dict_loader = LoadImaged(keys=("image", "label"))
    # data_dict = dict_loader(eval(one_data_path_dict))
    # # image shape: T2a(384, 384, 32)  ADC(256, 256, 32)  conT1a(432, 432, 32) fsT2a(400, 400, 32)
    # ds = get_data_base_transform()(data_dict)
    # # ds = Dataset(data=data_dict, transform=get_data_train_transform())
    # print('image shape:', ds['image'].shape)
    ds = Dataset(data=data_path_dict, transform=val_transforms)
    # batch每次处理两张，会显示在第一维度的通道数上。另：1个gpu`
    # TODO:collate_fn??
    # return DataLoader(dataset=ds, batch_size=4, num_workers=1)
    return monai.data.DataLoader(dataset=ds, batch_size=1, num_workers=1, collate_fn=list_data_collate, )


def pre_train_dataset():
    data_path_dict = []
    with open('./traindata_path_dict.txt', 'r') as f:
        # with open('./traindata_path_dict.txt', 'r') as f:
        # 共672次循环，4nii*168个人
        for one_data_path_dict in tqdm.tqdm(f.readlines()):
            print(one_data_path_dict)
            data_path_dict.append(one_data_path_dict)
    ds = Dataset(data=data_path_dict, transform=train_transforms)
    # dict_loader = LoadImaged(keys=("image", "label"))
    # data_dict = dict_loader(list(data_path_dict))
    # image shape: T2a(384, 384, 32)  ADC(256, 256, 32)  conT1a(432, 432, 32) fsT2a(400, 400, 32)
    # ds = get_data_train_transform()(data_dict)
    # ds = Dataset(data=data_dict, transform=get_data_train_transform())
    # print('image shape:', ds['image'].shape)
    # batch每次处理两张，会显示在第一维度的通道数上。另：1个gpu`
    # TODO:collate_fn??
    # loader = DataLoader(dataset=ds, batch_size=4, num_workers=1)
    # print(loader)
    dl = monai.data.DataLoader(dataset=ds, batch_size=2, shuffle=True, num_workers=1, collate_fn=list_data_collate,
                               pin_memory=torch.cuda.is_available(), )
    return dl


pre_train_dataset()


# *****************************************************  test
# *******************************************************************
def test():
    dict1 = {'4': 'Runoob', 'Age': 7, 'Name': '小菜鸟'}
    dict2 = {'1', 2}
    a = list(dict1).append(str(dict2))
    print(a)


# test()


def test2():
    image = LoadImaged(keys=['image', 'label'])({'image': 'D:\\code\\Shanxi_data\\data\\53\\149conT1a_RAW.nii',
                                                 'label': 'D:\\code\\Shanxi_data\\data\\53\\149conT1a_Merge.nii'})
    print(image)
    # 创建一个resize的方法
    # resize = Resized(keys=['image', 'label'], spatial_size=(400, 400, 24))
    # 对刚才添加维度后的数据进行resize操作
    # data_dict_resize = resize(dict_loader)
    data_dict = [{'image': 'D:\\code\\Shanxi_data\\data\\53\\149conT1a_RAW.nii',
                  'label': 'D:\\code\\Shanxi_data\\data\\53\\149conT1a_Merge.nii'}]
    # t2 = transforms.EnsureChannelFirstd(keys=['image', 'label'])(image)
    # t3 = transforms.Resized(keys=['image', 'label'], spatial_size=(400, 400, 32))(t2)
    # print('shape', t3['image'].shape)
    transform = Compose(
        [
            # LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            Resized(keys=['image', 'label'], spatial_size=(400, 400, 24)),
            transforms.CropForegroundd(keys=["image", 'label'], source_key='image'),

            # spatial aug
            transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),

            # intensity aug
            transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
            transforms.RandGaussianSmoothd(
                keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),
            ToTensord(keys=['image', 'label']),  # 把图像转成tensor格式
        ])
    ds = transform(image)
    # ds = Dataset(data=image, transform=transforms)
    # train_loader = DataLoader(
    #     ds, batch_size=4, shuffle=True, num_workers=1)
    print(f"image shape:", ds['image'].shape)
    print(f"label shape:", ds["label"].shape)

    plt.subplot(1, 2, 1)
    plt.title("raw_image")
    plt.imshow(image['image'][:, :, 5], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("transform_image")
    plt.imshow(ds['image'][0, :, :, 5], cmap='gray')
    plt.show()


# test2()

# *****************************************************  assemble data_path_dict.txt
# *******************************************************************

# 组装4个模态的图像和掩码的文件名字到8个列表/组装4个模态的图像和掩码到2个列表
def assemble_data():
    # if self.root is None:
    #     root_dir = r'D:\code\Shanxi_data\data'
    # else:
    #     root_dir = self.root
    # 168个人
    root_dir = r'D:\code\Shanxi_data\data'
    file_lists = os.listdir(root_dir)
    print(file_lists)
    # length = len(file_lists)
    images = []
    masks = []
    adc_images = []
    cont1a_images = []
    fst2a_images = []
    t2a_images = []

    adc_mask = []
    cont1a_mask = []
    fst2a_mask = []
    t2a_mask = []
    for single_name in file_lists:
        # single_list ——>   D:\code\Shanxi_data\data\0
        single_list = os.listdir(root_dir + './' + single_name)
        single_list_nii = list(filter(lambda x: '.nii' in x, single_list))
        for single_slice_nii in single_list_nii:
            image_str = join(root_dir, single_name, single_slice_nii)
            if 'merge' in single_slice_nii.lower():
                # if 'adc' in single_slice_nii.lower():
                #     adc_images.append(single_slice_nii)
                # elif 'cont1a' in single_slice_nii.lower():
                #     cont1a_images.append(single_slice_nii)
                # elif 'fst2a' in single_slice_nii.lower():
                #     fst2a_images.append(single_slice_nii)
                # else:
                #     t2a_images.append(single_slice_nii)
                # example ——>  D:\code\Shanxi_data\data\0\37fsT2a_RAW.nii
                # 上面是分模态列表 下面是合并模态列表，经检查，合并之后的image和mask也能对应
                masks.append(image_str)

            else:
                # if 'adc' in single_slice_nii.lower():
                #     adc_mask.append(single_slice_nii)
                # elif 'cont1a' in single_slice_nii.lower():
                #     cont1a_mask.append(single_slice_nii)
                # elif 'fst2a' in single_slice_nii.lower():
                #     fst2a_mask.append(single_slice_nii)
                # else:
                #     t2a_mask.append(single_slice_nii)
                images.append(image_str)
    # print('adc_images', adc_images)
    # print('cont1a_images', cont1a_images)
    # print('fst2a_images', fst2a_images)
    # print('t2a_images', t2a_images)
    # print('images', images)
    # print('masks', masks)

    # 组装每一个切片和掩码对应的字典，此时4种模态已经融合了
    data_path_dict = [{'image': image, 'label': label} for image, label in zip(images, masks)]
    # np.savetxt('./traindata_path_dict.txt', data_dict, fmt="%s")
    return data_path_dict

# assemble_data()
# import torch
# print(torch.cuda.is_available())

# print(torch.__version__)
