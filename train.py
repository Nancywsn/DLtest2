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
from evalute import get_confusion_matrix_for_3d


if __name__ == "__main__":
    IMG_SIZE = 128
    print ("=> the training size is {}".format(IMG_SIZE))

    train_loader = DataLoader(TinySegData(img_size=IMG_SIZE, phase='train'), batch_size=32, shuffle=True, num_workers=8)
    #val_loader = DataLoader(TinySegData(phase='val'), batch_size=1, shuffle=False, num_workers=0)

    model = PSPNet(n_classes=6, pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    mkdirs = lambda x: os.makedirs(x, exist_ok=True)
    # model.load_state_dict(torch.load("ckpt_seg/epoch_79_iou0.88.pth"))

    ckpt_dir = "ckpt_seg"
    mkdirs(ckpt_dir)
    epoch = 100

    for i in range(0, epoch):
        # train
        model.train()
        epoch_iou = []
        epoch_start = time.time()
        for j, (images, seg_gts, rets) in enumerate(train_loader):
            images = images.cuda()
            seg_gts = seg_gts.cuda()
            optimizer.zero_grad()

            seg_logit = model(images)
            loss_seg = criterion(seg_logit, seg_gts.long())
            loss = loss_seg
            loss.backward()
            optimizer.step()

            # epoch_loss += loss.item()
            if j % 10 == 0:
                seg_preds = torch.argmax(seg_logit, dim=1)
                seg_preds_np = seg_preds.detach().cpu().numpy()
                seg_gts_np = seg_gts.cpu().numpy()

                confusion_matrix = get_confusion_matrix_for_3d(seg_gts_np, seg_preds_np, class_num=6)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IU = IU_array.mean()

                log_str = "[E{}/{} - {}] ".format(i, epoch, j)
                log_str += "loss[seg]: {:0.4f}, miou: {:0.4f}, ".format(loss_seg.item(), mean_IU)
                print (log_str)

                images_np = np.transpose((images.cpu().numpy()+1)*127.5, (0, 2, 3, 1))
                n, h, w, c = images_np.shape
                images_np = images_np.reshape(n*h, w, -1)[:, :, 0]
                seg_preds_np = seg_preds_np.reshape(n*h, w)
                visual_np = np.concatenate([images_np, seg_preds_np*40], axis=1)       # NH * W
                cv2.imwrite('visual.png', visual_np)
                epoch_iou.append(mean_IU)

        epoch_iou = np.mean(epoch_iou)
        epoch_end = time.time()
        epoch_time = round(epoch_end-epoch_start, 2)
        print ("=> This epoch costs {}s...".format(epoch_time))
        if i % 10 == 0 or i ==  epoch-1:
            print ("=> saving to {}".format("{}/epoch_{}_iou{:0.2f}.pth".format(ckpt_dir, i, epoch_iou)))
            torch.save(model.state_dict(), "{}/epoch_{}_iou{:0.2f}.pth".format(ckpt_dir, i, epoch_iou))

