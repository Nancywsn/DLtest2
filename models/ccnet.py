import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone_8stride import resnet18

def INF(B, H, W):
    """
    构造一个形状为 [B*W, H, H] 的对角矩阵，其对角线值为 -inf，
    用于 CrissCrossAttention 中消除自注意力中的自身影响。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
 
        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
 
    def forward(self, x):
        b, _, h, w = x.size()
        # [b, c', h, w]
        query = self.ConvQuery(x)
        # 横向注意力分支：将 w 维度提到 batch 维度
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        # 纵向注意力分支
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        
        # Key 分支
        key = self.ConvKey(x)
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        
        # Value 分支
        value = self.ConvValue(x)
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        
        # 计算横向注意力 energy
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        # 计算纵向注意力 energy
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
        # 拼接能量，并在最后一个维度上做 softmax
        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))
        
        # 分离出两个注意力分支
        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)
 
        # 利用注意力对 value 进行加权
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
 
        return self.gamma * (out_H + out_W) + x

class RCCAModule(nn.Module):
    def __init__(self, recurrence=2, in_channels=2048, num_classes=33):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.inter_channels = in_channels // 4
        # 降维模块，将高维特征映射到较低的维度
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels)
        )
        self.CCA = CrissCrossAttention(self.inter_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False)
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.in_channels + self.inter_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            # nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            # nn.Upsample(size=(H, W), mode="bilinear", align_corners=True),
            # nn.Conv2d(self.inter_channels, self.num_classes, kernel_size=1)
        )

        self.cls_seg2 = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.num_classes, kernel_size=1)
        )
 
    def forward(self, x,H, W):
        output = self.conv_in(x)
        for i in range(self.recurrence):
            output = self.CCA(output)
        output = self.conv_out(output)
        # print(output.size()) #[1, 128, 32, 49]
        output = self.cls_seg(torch.cat([x, output], 1)) #[1, 6, 256, 392]

        # 动态上采样
        upsample = nn.Upsample(size=(H, W), mode="bilinear", align_corners=True)
        output = upsample(output)

        output = self.cls_seg2(output)

        return output

class CCNet(nn.Module):
    def __init__(self, num_classes):
        super(CCNet, self).__init__()
        # 使用 torchvision 提供的 resnet50 作为骨干网络，
        # 注意这里使用 backbone_8stride 中的 resnet50（不支持额外的 replace_stride_with_dilation）
        self.backbone = resnet18(pretrained=True)
        self.decode_head = RCCAModule(recurrence=2, in_channels=512, num_classes=num_classes)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda: self.cuda()
    
    def forward(self, x):
        # print(x.size()) #[1, 3, 256, 392]
        _, _, H, W = x.size()
        if self.use_cuda:
            x = x.cuda()
        
        # 提取特征：采用 resnet50 的前向传播过程，去掉最后的全局池化和全连接层
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # print(x.size())         #[1, 512, 32, 49]
        # 进行 RCCA 解码
        x = self.decode_head(x,H, W)
        # print(x.size()) #[1, 6, 256, 392]
        return x