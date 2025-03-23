import torch
from torch import nn, Tensor
from torch.nn import functional as F
from collections import OrderedDict
from typing import Dict, List
from models.backbone_8stride import resnet18  # 直接使用外部实现的 resnet18

# 提取骨干网络中指定层的输出
class IntermediateLayerGetter(nn.Module):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        super().__init__()
        # 检查 return_layers 的键是否存在于 model 的子模块中
        if not set(return_layers).issubset({name for name, _ in model.named_children()}):
            raise ValueError("Some keys in return_layers are not present in model")
        self.return_layers = return_layers.copy()
        layers = OrderedDict()
        # 仅保留到所有指定层均已被找到为止
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        self.layers = nn.ModuleDict(layers)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.layers.items():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out

# DeepLabV3 主模型（修改后的 forward 仅返回主输出张量）
class DeepLabV3(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: nn.Module = None):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        main_out = self.classifier(features["out"])
        out = F.interpolate(main_out, size=input_shape, mode='bilinear', align_corners=False)
        return out

# 辅助分类器，用于深度监督（若需要可在训练时额外使用）
class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]
        super().__init__(*layers)

# ASPP 中使用带膨胀卷积的模块
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super().__init__(*layers)

# ASPP 中使用自适应池化的分支
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        layers = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super().__init__(*layers)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# ASPP 模块：并行采集不同感受野信息后融合
class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: Tensor) -> Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

# 分类头，将 ASPP 模块和后续卷积融合后输出预测
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int):
        layers = [
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        ]
        super().__init__(*layers)

# 工厂函数：构造基于 resnet18 的 DeepLabV3 模型
def deeplabv3_resnet18(aux: bool = True, num_classes: int = 6, pretrain_backbone: bool = True) -> nn.Module:
    # 调用外部实现的 resnet18，设置 replace_stride_with_dilation 以获得 8 倍下采样
    backbone = resnet18(pretrained=pretrain_backbone)
    # 对于 resnet18：layer4 输出通道数为 512，layer3 输出通道数为 256
    out_inplanes = 512
    aux_inplanes = 256
    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers)
    # 此处 aux_classifier 虽然构造了，但在 forward 中未使用，因为我们只返回主输出
    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
    classifier = DeepLabHead(out_inplanes, num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    if torch.cuda.is_available():
        model = model.cuda()
    return model

