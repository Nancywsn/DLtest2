import torch.nn.functional as F
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import os, time
import random
random.seed(0)

from models.pspnet import PSPNet


def shortside_resize(image, mask=None, size=256):
    h, w = image.shape[:2]
    if h >= w:
        new_w = size
        new_h = int(h * (size*1.0/w))
        if new_h % 32 != 0:
            new_h = new_h + 32 - new_h % 32
    else:
        new_h = size
        new_w = int(w * (size*1.0/h))
        if new_w % 32 != 0:
            new_w = new_w + 32 - new_w % 32
    new_size = (int(new_w), int(new_h))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        return resized_image, resized_mask
    else:
        return resized_image

def overlay_mask(image, mask, countour_value=128, alpha=0.5):
    from scipy.ndimage import binary_erosion, binary_dilation

    image, dtype = image.copy(), image.dtype
    label_colours = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [255, 255, 0], [0, 0, 255], 
                     [255, 0, 255], [0, 255, 255], [128, 128, 128], [64, 0, 0], 
                     [191, 0, 0], [64, 128, 0], [191, 128, 0], [64, 0, 128], 
                     [191, 0, 128], [64, 128, 128], [191, 128, 128], [0, 64, 0], 
                     [128, 64, 0], [0, 191, 0], [128, 191, 0], [0, 64, 128], [128, 64, 128]]

    indices = np.unique(mask)
    for cls_index in indices:
        if cls_index != 0:
            mask_index = mask == cls_index
            cls_color = label_colours[cls_index]
            image[mask_index, :] = image[mask_index, :]*alpha + np.array(cls_color)[::-1]*(1-alpha)
            countours = binary_dilation(mask_index) ^ mask_index
            image[countours, :] = countour_value
    return image.astype(dtype)

if __name__ == "__main__":
    ## test
    model_path = "ckpt_seg/1_best_model.pth"
    image_path = "TinySeg/JPEGImages/06000.jpg"

    palette = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128], [128, 128, 0],
            [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64],
            [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128]]
    palette_vec = []
    for x in palette:
        palette_vec += x

    model = PSPNet(n_classes=6)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image_np = cv2.imread(image_path)
    image_np = shortside_resize(image_np)

    image = image_np / 127.5 - 1
    image = np.transpose(image, (2, 0, 1))

    cls_map = ['background', 'person', 'cat', 'plane', 'car', 'bird']

    image_th = torch.from_numpy(image).float().unsqueeze(0)
    with torch.no_grad():
        seg_logit = model(image_th)
    seg_pred = torch.argmax(seg_logit, dim=1)[0].cpu().numpy()
    seg_pred = seg_pred.astype(np.uint8)

    seg_visual = Image.fromarray(seg_pred)
    seg_visual.putpalette(palette_vec)
    seg_visual = seg_visual.convert('RGB')
    seg_visual = np.asarray(seg_visual)

    img_visual_mask = overlay_mask(image_np, seg_pred)
    comb_res = np.concatenate([image_np, seg_visual, img_visual_mask], axis=1)

    os.makedirs("outputs1", exist_ok=True)
    image_name = image_path.split('/')[-1]

    cv2.imwrite('outputs/{}'.format(image_name), comb_res)

    