from model.MGFNet import MGFNet
import copy
import datetime
import os
import pandas as pd
import torchvision
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet34_Weights


if __name__ == "__main__":
    from thop import profile, clever_format

    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 1, 256, 256)
    net = MGFNet(num_class=6)
    # net.load_state_dict(torch.load('../checkpoint/baseline_resnet50_epoch_33_91.46417727444361'), strict=False)
    flops, params = profile(net, inputs=(x, y))
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPs:", macs)
    print("params:", params)
