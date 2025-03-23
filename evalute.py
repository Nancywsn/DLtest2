import torch.nn.functional as F
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import numpy as np
import os, time
import random
random.seed(0)

from models.pspnet import PSPNet
from dataloader import TinySegData
import seaborn as sns
import matplotlib.pyplot as plt

################################# FUNCTIONS #################################
def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the number of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')

        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def get_confusion_matrix_for_3d(gt_label, pred_label, class_num):
    confusion_matrix = np.zeros((class_num, class_num))

    for sub_gt_label, sub_pred_label in zip(gt_label, pred_label):
        sub_gt_label = sub_gt_label[sub_gt_label != 255]
        sub_pred_label = sub_pred_label[sub_pred_label != 255]
        cm = get_confusion_matrix(sub_gt_label, sub_pred_label, class_num)
        confusion_matrix += cm
    return confusion_matrix


if __name__ == "__main__":
    ## test
    model_path = "ckpt_seg/1_best_model.pth"
    image_path = "TinySeg/JPEGImages/00000.jpg"
    classes = 6
    
    # Define class names
    CLASSES = ['background', 'person', 'cat', 'plane', 'car', 'bird']

    model = PSPNet(n_classes=classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()

    testloader = DataLoader(TinySegData(phase='val'), batch_size=1, shuffle=False, num_workers=0)

    data_list = []
    val_iou = []
    val_pixel_accuracy= []
    confusion_matrix = np.zeros((classes,classes))

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    with torch.no_grad():
        for images, seg_gts, sets in testloader:
            if torch.cuda.is_available():
                images, seg_gts = images.cuda(), seg_gts.cuda()
            # print(images.shape) #torch.Size([1, 3, 256, 317])
            # print(seg_gts.shape) #torch.Size([1, 256, 317])

            seg_logit = model(images)
            seg_preds = torch.argmax(seg_logit, dim=1)
            seg_preds_np = seg_preds.detach().cpu().numpy()
            seg_gts_np = seg_gts.cpu().numpy()

            confusion_matrix_result = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=classes)
            confusion_matrix += confusion_matrix_result

            # 计算像素准确率
            correct_pixels = np.sum(seg_preds_np == seg_gts_np)
            total_pixels = seg_gts_np.size
            pixel_accuracy = correct_pixels / total_pixels
            val_pixel_accuracy.append(pixel_accuracy)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png')

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()
    print(mean_IU)

    val_pixel_accuracy = np.mean(val_pixel_accuracy)
    print(f"Eval: Validation mIoU: {mean_IU:.4f}, Pixel Accuracy: {val_pixel_accuracy:.4f}")
