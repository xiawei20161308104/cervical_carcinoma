import logging
import os
import sys
import tempfile
from glob import glob
from os.path import join

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd, Resized, MapTransform,
)
from monai.visualize import plot_2d_or_3d_image


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    如T2加权成像（T2WI）label t2wi 序列能有效显示宫颈癌的病灶形态、浸润深度等；
    增强T1加权成像（cT1WI）label ct1wi 序列则会强化病灶区域与正常组织的对比[22]，
    表观弥散系数（ADC）label adc 序列在表现肿瘤密度与侵袭性上有一定的优势[23],
    抑脂T2加权成像（T2WI_FS）label t2wifs 序列对病变区域的炎症程度和水肿程度较为敏感。

    DWI和ADC是可以转化，他俩正好是反的，DWI是高信号的时候ADC是低信号
    cT1WI_ADC_T2WI、cT1WI_ADC_T2WI-FS最好

    Convert labels to multi channels based on brats classes:
    label 1 is the t2wi
    label 2 is the ct1wi
    label 3 is the ct1wi_adc_t2wifs

    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).(xv:TODO i dont know)

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = [torch.logical_or(d[key] == 2, d[key] == 3),
                      torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1), d[key] == 2]
            # merge label 2 and label 3 to construct TC
            # merge labels 1, 2 and 3 to construct WT
            # label 2 is ET
            d[key] = torch.stack(result, axis=0).float()
        return d


def main():
    train_files, val_files = assemble_data()

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
            # Resized(keys=['img', 'seg'], spatial_size=(384, 384, 32), allow_missing_keys=True),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 32], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 1)),
            ConvertToMultiChannelBasedOnBratsClassesd()
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
        ]
    )


if __name__ == "__main__":
    main()


def assemble_data():
    # 168个人
    root_dir = r'D:\code\Shanxi_data\data'
    file_lists = os.listdir(root_dir)
    images = []
    masks = []
    for single_name in file_lists:
        # single_list ——>   D:\code\Shanxi_data\data\0
        single_list = os.listdir(root_dir + './' + single_name)
        single_list_nii = list(filter(lambda x: '.nii' in x, single_list))
        for single_slice_nii in single_list_nii:
            image_str = join(root_dir, single_name, single_slice_nii)
            if 'merge' in single_slice_nii.lower():
                # 上面是分模态列表 下面是合并模态列表，经检查，合并之后的image和mask也能对应
                masks.append(image_str)

            else:
                images.append(image_str)

    # 组装每一个切片和掩码对应的字典，此时4种模态已经融合了
    train_files = [{'img': img, 'seg': seg} for img, seg in zip(images[:148], masks[:148])]
    val_files = [{'img': img, 'seg': seg} for img, seg in zip(images[-20:], masks[-20:])]
    return train_files, val_files
