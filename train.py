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
import torch.optim.lr_scheduler as lr_scheduler

from models.ccnet import CCNet
from models.pspnet import PSPNet
from models.deeplabv3 import deeplabv3_resnet18
from dataloader import TinySegData
from evalute import get_confusion_matrix_for_3d,get_confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import matplotlib.pyplot as plt
import pickle

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 保存随机种子的状态
state_dict = {
    'torch': torch.get_rng_state(),
    'numpy': np.random.get_state(),
    'random': random.getstate()
}


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
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--model', type=str, default='pspnet', choices=['pspnet', 'ccnet', 'deeplabv3'], help='Model to use for training')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--tmax', type=int, default=20, help='Maximum number of iterations for CosineAnnealingLR')
    parser.add_argument('--etamin', type=float, default=1e-5, help='Minimum learning rate for CosineAnnealingLR')
    parser.add_argument('--outputdir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--aug', type=str, default='advanced', choices=['none', 'basic', 'advanced'], help='Type of data augmentation to use')
    args = parser.parse_args()

    IMG_SIZE = 128
    print ("=> the training size is {}".format(IMG_SIZE))
    classes=6
    CLASSES = ['background', 'person', 'cat', 'plane', 'car', 'bird']
    epoch = 100
    best_miou=0
    early_stop_patience = 10
    best_model_name = args.model+ "_" + args.aug +"_best_model.pth"
    early_model_name = args.model+ "_" + args.aug+"_early_model.pth"

    mkdirs = lambda x: os.makedirs(x, exist_ok=True)
    ckpt_dir = "ckpt_seg"
    mkdirs(ckpt_dir)
    output_dir = args.outputdir + "_" + args.model + "_" + args.aug
    mkdirs(output_dir)
    csv_path = os.path.join(output_dir, "metrics.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # 随机种子保存到文件
    seed_path = os.path.join(output_dir, "random_seeds.pkl")
    with open(seed_path, 'wb') as f:
        pickle.dump(state_dict, f)

    train_loader = DataLoader(TinySegData(img_size=IMG_SIZE, phase='train',aug=args.aug), batch_size=args.batchsize, shuffle=True, num_workers=8)
    val_loader = DataLoader(TinySegData(phase='val'), batch_size=1, shuffle=False, num_workers=0)

    if args.model == 'pspnet':
        model = PSPNet(n_classes=classes, pretrained=True)
    elif args.model == 'ccnet':
        model = CCNet(num_classes=classes)
    elif args.model == 'deeplabv3':
        model = deeplabv3_resnet18(num_classes=classes)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=args.etamin)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(num_classes=classes)

    best_epoch_metrics={}
    for i in range(0, epoch):
        # train
        model.train()
        epoch_iou = []
        epoch_loss=[]

        epoch_start = time.time()
        for j, (images, seg_gts, rets) in enumerate(train_loader):
            images = images.cuda() #torch.Size([32, 3, 128, 128])
            seg_gts = seg_gts.cuda() #torch.Size([32, 128, 128])
            # print(images.shape)
            # print(seg_gts.shape)

            optimizer.zero_grad()
            seg_logit = model(images)

            # print(seg_logit.shape) #torch.Size([64, 6, 128, 128])
            # print(seg_gts.shape) #torch.Size([64, 128, 128])
            loss_ce = criterion_ce(seg_logit, seg_gts.long())
            loss_dice = criterion_dice(seg_logit, seg_gts.long())
            loss_seg = loss_ce + loss_dice
            epoch_loss.append(loss_seg.item())

            loss = loss_seg
            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                seg_preds = torch.argmax(seg_logit, dim=1)
                seg_preds_np = seg_preds.detach().cpu().numpy() #(32, 128, 128)
                seg_gts_np = seg_gts.cpu().numpy() #(32, 128, 128)
                # print(seg_preds_np.shape)
                # print(seg_gts_np.shape)

                # 计算混淆矩阵，基于混淆矩阵计算每个类别的交并比（IoU）和平均交并比（mean IoU）
                confusion_matrix_result = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=classes)
                pos = confusion_matrix_result.sum(1)
                res = confusion_matrix_result.sum(0)
                tp = np.diag(confusion_matrix_result)
                IU_array = (tp / np.maximum(1.0, pos + res - tp))
                # print(IU_array) #[0.81807475 0.60226701 0.70165469 0.25323819 0.5943455  0.        ]
                # print(IU_array.size) #6
                mean_IU = IU_array.mean()
                # print(mean_IU) #0.49493002184574514

                # plt.figure(figsize=(10, 8))
                # sns.heatmap(confusion_matrix_result, annot=True, fmt=".2f", xticklabels=CLASSES, yticklabels=CLASSES)
                # plt.xlabel('Predicted')
                # plt.ylabel('True')
                # plt.title('confusion_matrix_result')
                # plt.savefig(os.path.join(output_dir, "confusion_matrix_result.png"))

                log_str = "[E{}/{} - {}] ".format(i+1, epoch, j)
                log_str += "loss[seg]: {:0.4f}, miou: {:0.4f}, lr: {:0.6f}".format(loss_seg.item(), mean_IU, optimizer.param_groups[0]['lr'])
                print(log_str)

                # # 实时可视化预测结果
                # images_np = np.transpose((images.cpu().numpy()+1)*127.5, (0, 2, 3, 1)).astype(np.uint8)
                # n, h, w, c = images_np.shape
                # images_np = images_np.reshape(n * h, w, c)
                # seg_preds_np = seg_preds_np.reshape(n*h, w)
                # seg_preds_color = cv2.applyColorMap((seg_preds_np * 40).astype(np.uint8), cv2.COLORMAP_JET)
                # visual_np = np.concatenate([images_np,seg_preds_color], axis=1)       # NH * W
                # cv2.imwrite(os.path.join(output_dir, "train_visual.png"), visual_np)

                epoch_iou.append(mean_IU)

        epoch_iou = np.mean(epoch_iou)
        print(f"Train:  Epoch {i+1} Loss: {np.mean(epoch_loss):.4f}, mIoU: {epoch_iou:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 验证阶段
        model.eval()
        val_iou = []
        loss_eval=[]
        dice = []
        val_pixel_accuracy= []
        confusion = np.zeros((classes,classes))
        
        # for j, (images, seg_gts, rets) in enumerate(val_loader):
        for images, seg_gts, sets in val_loader:
            with torch.no_grad():
                images, seg_gts = images.cuda(), seg_gts.cuda()

                seg_logit = model(images)
                seg_preds = torch.argmax(seg_logit, dim=1)
                seg_preds_np = seg_preds.detach().cpu().numpy()
                seg_gts_np = seg_gts.cpu().numpy()

                loss_ce = criterion_ce(seg_logit, seg_gts.long()) 
                loss_dice = criterion_dice(seg_logit, seg_gts.long())
                loss = loss_ce + loss_dice
                loss_eval.append(loss.item())
                dice.append(loss_dice.item())

                confusion_matrix_result = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=classes)
                confusion += confusion_matrix_result

                # 计算像素准确率
                correct_pixels = np.sum(seg_preds_np == seg_gts_np)
                total_pixels = seg_gts_np.size
                pixel_accuracy = correct_pixels / total_pixels
                val_pixel_accuracy.append(pixel_accuracy)

                # # 实时可视化预测结果
                # images_np = np.transpose((images.cpu().numpy()+1)*127.5, (0, 2, 3, 1)).astype(np.uint8)
                # n, h, w, c = images_np.shape
                # images_np = images_np.reshape(n * h, w, c)
                # seg_preds_np = seg_preds_np.reshape(n*h, w)
                # seg_preds_color = cv2.applyColorMap((seg_preds_np * 40).astype(np.uint8), cv2.COLORMAP_JET)
                # visual_np = np.concatenate([images_np,seg_preds_color], axis=1)       # NH * W
                # cv2.imwrite(os.path.join(output_dir, "eval_visual.png"), visual_np)

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion / confusion.sum(axis=1, keepdims=True) * 100, annot=True, fmt=".2f", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (%)')
        plt.savefig(os.path.join(output_dir, "confusion.png"))

        pos = confusion.sum(1)
        res = confusion.sum(0)
        tp = np.diag(confusion)
        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        print(mean_IU)
        val_iou = mean_IU
        val_pixel_accuracy = np.mean(val_pixel_accuracy)    
        loss_eval = np.mean(loss_eval)
        dice = np.mean(dice)     
        print(f"Eval: Validation mIoU: {val_iou:.4f}, Validation loss: {loss_eval:.4f}, Pixel Accuracy: {val_pixel_accuracy:.4f}")
        
        epoch_end = time.time()
        epoch_time = round(epoch_end-epoch_start, 2)
        print ("=> This epoch costs {}s...".format(epoch_time))

        epoch_metrics = {
            'epoch': i+1,
            'train_loss': np.mean(epoch_loss),
            'val_loss': loss_eval,
            'val_miou': val_iou,
            'pixel_accuracy': val_pixel_accuracy,
            'dice': dice
        }
        # 保存每个epoch的损失和mIoU到csv文件
        with open(csv_path, "a") as f:
            if i == 0:
                f.write("epoch,train_loss,val_loss,val_miou,pixel_accuracy,dice\n")
            f.write(f"{i+1},{np.mean(epoch_loss):.4f},{loss_eval:.4f},{val_iou:.4f},{val_pixel_accuracy:.4f},{dice:.4f}\n")
           
        # 早停条件
        if val_iou > best_miou:
            best_miou = val_iou
            no_improvement_epochs = 0
            best_epoch_metrics = epoch_metrics
            torch.save(model.state_dict(), os.path.join(ckpt_dir, best_model_name))
        else:
            no_improvement_epochs += 1        

        if no_improvement_epochs == early_stop_patience:
            print(f"Early stopping at epoch {i+1} due to no improvement in validation mIoU for {early_stop_patience} epochs.")
            print(f"early_stop Best epoch metrics: {best_epoch_metrics}")
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{i+1}_{early_model_name}"))
            # break
        print("no_improvement_epochs:"+str(no_improvement_epochs))

        # 在每个 epoch 结束时更新学习率
        scheduler.step()

    print(f"Best epoch metrics: {best_epoch_metrics}")
    
    draw_metrics(csv_path, output_dir)

