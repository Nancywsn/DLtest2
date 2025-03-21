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
import torch.optim.lr_scheduler as lr_scheduler

from models.pspnet import PSPNet
from dataloader import TinySegData
from evalute import get_confusion_matrix_for_3d,get_confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)

        return 1 - dice.mean()
    
def draw_metrics(csv_path, output_dir):

    metrics = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(metrics['epoch'], metrics['val_miou'], label='Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.title('mIoU Curve')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics.png"))
    plt.close()


if __name__ == "__main__":
    IMG_SIZE = 128
    print ("=> the training size is {}".format(IMG_SIZE))
    classes=6
    epoch = 50
    best_miou=0
    early_stop_patience = 10

    train_loader = DataLoader(TinySegData(img_size=IMG_SIZE, phase='train'), batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(TinySegData(phase='val'), batch_size=1, shuffle=False, num_workers=0)

    model = PSPNet(n_classes=classes, pretrained=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(num_classes=classes)


    mkdirs = lambda x: os.makedirs(x, exist_ok=True)
    # model.load_state_dict(torch.load("ckpt_seg/epoch_79_iou0.88.pth"))
    ckpt_dir = "ckpt_seg"
    mkdirs(ckpt_dir)
    output_dir = "output"
    mkdirs(output_dir)
    csv_path = os.path.join(output_dir, "metrics.csv")    


    for i in range(0, epoch):
        # train
        model.train()
        epoch_iou = []
        epoch_loss=[]

        epoch_start = time.time()
        for j, (images, seg_gts, rets) in enumerate(train_loader):
            images = images.cuda()
            seg_gts = seg_gts.cuda()

            optimizer.zero_grad()
            seg_logit = model(images)

            # loss_seg = criterion(seg_logit, seg_gts.long())
            # loss = loss_seg

            loss_ce = criterion_ce(seg_logit, seg_gts.long())
            loss_dice = criterion_dice(seg_logit, seg_gts.long())
            loss_seg = loss_ce + loss_dice
            epoch_loss.append(loss_seg.item())

            loss = loss_seg
            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                seg_preds = torch.argmax(seg_logit, dim=1)
                seg_preds_np = seg_preds.detach().cpu().numpy()
                seg_gts_np = seg_gts.cpu().numpy()

                # 计算混淆矩阵，基于混淆矩阵计算每个类别的交并比（IoU）和平均交并比（mean IoU）
                confusion_matrix_result = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=classes)
                pos = confusion_matrix_result.sum(1)
                res = confusion_matrix_result.sum(0)
                tp = np.diag(confusion_matrix_result)
                IU_array = (tp / np.maximum(1.0, pos + res - tp))
                # print(IU_array)
                mean_IU = IU_array.mean()
                # print(mean_IU)

                log_str = "[E{}/{} - {}] ".format(i, epoch, j)
                log_str += "loss[seg]: {:0.4f}, miou: {:0.4f}, lr: {:0.6f}".format(loss_seg.item(), mean_IU, optimizer.param_groups[0]['lr'])
                print(log_str)

                # 实时可视化预测结果
                images_np = np.transpose((images.cpu().numpy()+1)*127.5, (0, 2, 3, 1))
                n, h, w, c = images_np.shape
                images_np = images_np.reshape(n*h, w, -1)[:, :, 0]
                seg_preds_np = seg_preds_np.reshape(n*h, w)
                visual_np = np.concatenate([images_np, seg_preds_np*40], axis=1)       # NH * W
                cv2.imwrite('visual.png', visual_np)
                epoch_iou.append(mean_IU)

        epoch_iou = np.mean(epoch_iou)
        print(f"Train:  Epoch {i} Loss: {np.mean(epoch_loss):.4f}, mIoU: {epoch_iou:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 验证阶段
        model.eval()
        val_iou = []
        loss_eval=[]
        with torch.no_grad():
            for images, seg_gts, sets in val_loader:
                if torch.cuda.is_available():
                    images, seg_gts = images.cuda(), seg_gts.cuda()

                seg_logit = model(images)
                seg_preds = torch.argmax(seg_logit, dim=1)
                seg_preds_np = seg_preds.detach().cpu().numpy()
                seg_gts_np = seg_gts.cpu().numpy()

                loss_ce = criterion_ce(seg_logit, seg_gts.long())
                loss_dice = criterion_dice(seg_logit, seg_gts.long())
                loss = loss_ce + loss_dice
                loss_eval.append(loss.item())

                # 计算混淆矩阵并归一化
                # cm = confusion_matrix(seg_gts_np.flatten(), seg_preds_np.flatten(), normalize='true')
                # # print(cm)
                # pos = cm.sum(1)
                # res = cm.sum(0)
                # tp = np.diag(cm)
                # IU_array = (tp / np.maximum(1.0, pos + res - tp))
                # mean_IU = IU_array.mean()
                # print(mean_IU)

                confusion_matrix_result = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=classes)
                pos = confusion_matrix_result.sum(1)
                res = confusion_matrix_result.sum(0)
                tp = np.diag(confusion_matrix_result)
                IU_array = (tp / np.maximum(1.0, pos + res - tp))
                print(IU_array.size)
                mean_IU = IU_array.mean()
                print(mean_IU)
                val_iou.append(mean_IU)


                # 计算像素准确率
                correct_pixels = np.sum(seg_preds_np == seg_gts_np)
                total_pixels = seg_gts_np.size
                pixel_accuracy = correct_pixels / total_pixels

        val_iou = np.mean(val_iou)
        loss_eval = np.mean(loss_eval)     
        print(f"Eval: Validation mIoU: {val_iou:.4f}, Validation loss: {loss_eval:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}")
        epoch_end = time.time()
        epoch_time = round(epoch_end-epoch_start, 2)
        print ("=> This epoch costs {}s...".format(epoch_time))
        # 保存每个epoch的损失和mIoU到csv文件
        with open(csv_path, "a") as f:
            if i == 0:
                f.write("epoch,train_loss,val_loss,val_miou,pixel_accuracy\n")
            f.write(f"{i},{np.mean(epoch_loss)},{loss_eval},{val_iou},{pixel_accuracy}\n")
       
        # 早停条件
        if val_iou > best_miou:
            best_miou = val_iou
            no_improvement_epochs = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"best_model.pth"))
        else:
            no_improvement_epochs += 1        
        if no_improvement_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {i} due to no improvement in validation mIoU for {early_stop_patience} epochs.")
            break
        # 在每个 epoch 结束时更新学习率
        scheduler.step()



