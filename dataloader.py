import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
import random

random.seed(0)

from albumentations import Compose, HorizontalFlip, Rotate, RandomBrightnessContrast, ElasticTransform, GridDistortion, RandomCrop, Resize


readvdnames = lambda x: open(x).read().rstrip().split('\n')

class TinySegData(Dataset):
    def __init__(self, db_root="TinySeg", img_size=256, phase='train',aug='none'):
        # classes = ['person', 'cat', 'plane', 'car', 'bird']
        # seg_ids = [1, 2, 3, 4, 5]

        templ_image = db_root + "/JPEGImages/{}.jpg"
        templ_mask = db_root + "/Annotations/{}.png"

        ids = readvdnames(db_root + "/ImageSets/" + phase + ".txt")

        # build training and testing dbs
        samples = []
        for i in ids:
            samples.append([templ_image.format(i), templ_mask.format(i)])
        self.samples = samples
        self.phase = phase
        self.db_root = db_root
        self.img_size = img_size
        self.aug = aug

        if not self.phase == 'train':
            print ("resize and augmentation will not be applied...")

        self.base_aug = Compose([
            HorizontalFlip(p=0.5),
            Rotate(limit=15, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3)
        ])

        self.advanced_aug = Compose([
            ElasticTransform(alpha=1, sigma=50, p=0.3),
            GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            RandomCrop(width=96, height=96, p=0.5),
            Resize(height=img_size, width=img_size, p=1.0)  # 上采样回原始尺寸
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.phase == 'train':
            return self.get_train_item(idx)
        else:
            return self.get_test_item(idx)

    def get_train_item(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample[0])

        image = np.asarray(image)[..., ::-1]     # to BGR

        # 打开分割标注图像，将其转换为调色板模式（P 模式），然后转换为 NumPy 数组，并将数据类型转换为 uint8
        seg_gt = (np.asarray(Image.open(sample[1]).convert('P'))).astype(np.uint8)

        image = image.astype(np.float32)
        image = image / 127.5 - 1        # -1~1

        #'none', 'basic', 'advanced'
        if self.aug == 'basic':
            # 基础数据增强
            # print (image.shape, seg_gt.shape)
            augmented = self.base_aug(image=image, mask=seg_gt)
            image = augmented['image']
            seg_gt = augmented['mask']
        elif self.aug == 'advanced':    
            # 基础数据增强
            # print (image.shape, seg_gt.shape)
            augmented = self.base_aug(image=image, mask=seg_gt)
            image = augmented['image']
            seg_gt = augmented['mask']
            # 高级数据增强
            augmented = self.advanced_aug(image=image, mask=seg_gt)
            image = augmented['image']
            seg_gt = augmented['mask']
            # print("advanced augmentation")



        # 如果要求的size不是256，则进行resize
        if self.img_size != 256:
            new_size = (self.img_size, self.img_size)
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            seg_gt = cv2.resize(seg_gt, new_size, interpolation=cv2.INTER_NEAREST)

        image = np.transpose(image, (2, 0, 1))      # To CHW

        return image, seg_gt, sample

    def get_test_item(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample[0])
        seg_gt = (np.asarray(Image.open(sample[1]).convert('P'))).astype(np.uint8)

        image = image.astype(np.float32)
        image = image / 127.5 - 1        # -1~1

        # 如果要求的size不是256，则进行resize
        if self.img_size != 256:
            new_size = (self.img_size, self.img_size)
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            seg_gt = cv2.resize(seg_gt, new_size, interpolation=cv2.INTER_NEAREST)

        image = np.transpose(image, (2, 0, 1))

        # sample = self.samples[idx]
        # image = cv2.imread(sample[0])
        # seg_gt = (np.asarray(Image.open(sample[1]).convert('P'))).astype(np.uint8)
        # image = image.astype(np.float32)
        # image = image / 127.5 - 1        # -1~1
        # image = np.transpose(image, (2, 0, 1))

        return image, seg_gt, sample
    
    
if __name__ == "__main__":
    classes=6
    CLASSES = ['background', 'person', 'cat', 'plane', 'car', 'bird']
    IMG_SIZE = 128

    val_loader = DataLoader(TinySegData(img_size=IMG_SIZE, phase='val'), batch_size=1, shuffle=False, num_workers=0)
    for i, (image, seg_gt, sample) in enumerate(val_loader):
        print (image.shape, seg_gt.shape, sample)

        # 将图像转换为 NumPy 数组并转换为 HWC 格式
        images_np = np.transpose((image.numpy() + 1) * 127.5, (0, 2, 3, 1)).astype(np.uint8)
        n, h, w, c = images_np.shape
        images_np = images_np.reshape(n * h, w, c)
        
        # 将标签转换为单通道图像并转换为三通道彩色图像
        seg_preds_np = seg_gt.numpy().reshape(n * h, w)
        seg_preds_color = cv2.applyColorMap((seg_preds_np * 40).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 拼接图像和标签
        visual_np = np.concatenate([images_np, seg_preds_color], axis=1)  # NH * W
        
        # 保存彩色图像
        cv2.imwrite('vi.png', visual_np)

        print("Segmentation Ground Truth Array:")
        print(seg_gt)
        print("Unique values in seg_gt:")
        print(np.unique(seg_gt))
 
        break