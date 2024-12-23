import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model_plus.fusion_transformer import TransformerBlock

# 原始版本
class MSGFM(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(MSGFM, self).__init__()

        #####################################  CWF weight d1 prediction  ############################################
        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.SPALayer1 = SPALayer(in_channels * 2)
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)

    def forward(self, input_rgb, input_dsm):
        ablation = False

        if ablation:
            fusion = torch.cat((input_dsm, input_rgb), dim=1)
            fusion = self.conv(fusion)
        else:
            d_1 = torch.cat((input_rgb, input_dsm), dim=1)
            d_1 = self.SPALayer1(d_1)
            weight_d1 = self.d1_weight_classifier(d_1)
            weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
            w_d1 = weight_d1
            w_d2 = 1 - weight_d1
            fusion = w_d1 * input_rgb + w_d2 * input_dsm

        return fusion


class SPALayer(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SPALayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool7 = nn.AdaptiveAvgPool2d(7)
        self.weight = nn.parameter.Parameter(torch.ones(1, 3, 1, 1, 1))
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            activation,
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool4(x)
        y4 = self.avg_pool7(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2, size=[7, 7]).unsqueeze(dim=1),
             F.interpolate(y1, size=[7, 7]).unsqueeze(dim=1)],
            dim=1)
        y = (y * self.weight).sum(dim=1, keepdim=False)
        y = self.transform(y)
        y = F.interpolate(y, size=x.size()[2:])

        return x * y

# 引入Non-local注意力版本
class fusion_cwf_MSFE(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(fusion_cwf_MSFE, self).__init__()

        # self.channel_emb = ChannelEmbed(in_channels=out_channels, out_channels=out_channels, reduction=reduction, norm_layer=norm_layer)
        #####################################  CWF weight d1 prediction  ############################################
        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.SPALayer1 = SPALayer(in_channels * 2)
        self.SPALayer2 = SPALayer(out_channels)
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)
        self.MSFE = MSFE(dim=in_channels, in_dim=in_channels)

    def forward(self, input_rgb, input_t):
        ablation = False

        if ablation:
            fusion = torch.cat((input_t, input_rgb), dim=1)
            fusion = self.conv(fusion)
        else:
            d_1 = torch.cat((input_rgb, input_t), dim=1)
            d_1 = self.SPALayer1(d_1)
            weight_d1 = self.d1_weight_classifier(d_1)
            weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
            w_d1 = weight_d1
            w_d2 = 1 - weight_d1
            out_rgb = self.MSFE(input_rgb)
            fusion = w_d1 * input_rgb + w_d2 * input_t + w_d2 * out_rgb
        return fusion

class fusion_cwf_SPA(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(fusion_cwf_SPA, self).__init__()

        #####################################  CWF weight d1 prediction  ############################################
        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.SPALayer_rgb = SPALayer(in_channels)
        self.SPALayer_dsm = SPALayer(in_channels)

        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)

    def forward(self, input_rgb, input_dsm):
        ablation = False

        if ablation:
            fusion = torch.cat((input_dsm, input_rgb), dim=1)
            fusion = self.conv(fusion)
        else:
            input_rgb = self.SPALayer_rgb(input_rgb)
            input_dsm = self.SPALayer_dsm(input_dsm)
            d_1 = torch.cat((input_rgb, input_dsm), dim=1)
            weight_d1 = self.d1_weight_classifier(d_1)
            weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
            w_d1 = weight_d1
            w_d2 = 1 - weight_d1
            fusion = w_d1 * input_rgb + w_d2 * input_dsm

        return fusion


class MSFE(nn.Module):
    def __init__(self, dim, in_dim):
        super(MSFE, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(3 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + conv2

        conv3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',
                              align_corners=True)

        return self.fuse(torch.cat((conv1, out2, conv3), 1))


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1), activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = MSFE(dim=channels_in, in_dim=channels_in)
        self.se_depth = SqueezeAndExcitation(channels_in, activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out

# 添加激励挤压
class fusion_cwf_se(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(fusion_cwf_se, self).__init__()

        #####################################  CWF weight d1 prediction  ############################################
        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.SPALayer1 = SPALayer(in_channels * 2)
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)
        # SE激励挤压
        self.se = SqueezeAndExcitation(channel=in_channels)

    def forward(self, input_rgb, input_dsm):
        ablation = False

        if ablation:
            fusion = torch.cat((input_dsm, input_rgb), dim=1)
            fusion = self.conv(fusion)
        else:
            d_1 = torch.cat((input_rgb, input_dsm), dim=1)
            d_1 = self.SPALayer1(d_1)
            weight_d1 = self.d1_weight_classifier(d_1)
            weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
            w_d1 = weight_d1
            w_d2 = 1 - weight_d1
            fusion = w_d1 * input_rgb + w_d2 * input_dsm
            fusion = self.se(fusion)

        return fusion



class SmallScaleAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SmallScaleAttentionModule, self).__init__()
        self.in_channels = in_channels
        reduction = 16

        # Define pooling layers
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        self.avg_pool4 = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)

        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=4, stride=4, padding=1)

        # Define convolution layers for attention
        self.conv_avg2 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_avg3 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_avg4 = nn.Conv2d(in_channels, in_channels // reduction, 1)

        self.conv_max2 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_max3 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_max4 = nn.Conv2d(in_channels, in_channels // reduction, 1)

        # Convolution to combine attention maps
        self.combine_conv = nn.Conv2d(in_channels // reduction * 6, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # Apply average pooling and convolution
        y_avg2 = F.interpolate(self.conv_avg2(self.avg_pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        y_avg3 = F.interpolate(self.conv_avg3(self.avg_pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        y_avg4 = F.interpolate(self.conv_avg4(self.avg_pool4(x)), size=(h, w), mode='bilinear', align_corners=True)

        # Apply max pooling and convolution
        y_max2 = F.interpolate(self.conv_max2(self.max_pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        y_max3 = F.interpolate(self.conv_max3(self.max_pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        y_max4 = F.interpolate(self.conv_max4(self.max_pool4(x)), size=(h, w), mode='bilinear', align_corners=True)

        # Concatenate the pooled features
        y = torch.cat([y_avg2, y_avg3, y_avg4, y_max2, y_max3, y_max4], dim=1)

        # Combine attention maps
        gama = self.sigmoid(self.combine_conv(y))

        # Apply attention
        out = x * gama
        return out


class SmallScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallScaleFeatureExtractor, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 标准卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)  # 空洞卷积
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)  # 空洞卷积
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x0 = self.conv0(x)  # 标准卷积
        x1 = self.conv1(x)  # 提取小尺度信息
        x2 = self.conv2(x)  # 提取中尺度信息
        x3 = self.conv3(x)  # 提取中尺度信息

        out = x0 + (x1 + x2 + x3) * self.gamma  # 融合多尺度信息
        return out

# 最优版本
class fusion_cwf_SmallScaleAttention(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(fusion_cwf_SmallScaleAttention, self).__init__()

        ################MultiScaleFeatureExtractor####################
        self.rgb_attention = SmallScaleAttentionModule(in_channels)
        self.rgb_feature_extractor = SmallScaleFeatureExtractor(in_channels, in_channels)
        #####################################  CWF weight d1 prediction  ############################################
        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.SPALayer1 = SPALayer(in_channels * 2)
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)

    def forward(self, input_rgb, input_dsm):
        ablation = False

        if ablation:
            fusion = torch.cat((input_dsm, input_rgb), dim=1)
            fusion = self.conv(fusion)
        else:
            d_1 = torch.cat((input_rgb, input_dsm), dim=1)
            d_1 = self.SPALayer1(d_1)
            weight_d1 = self.d1_weight_classifier(d_1)
            weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
            w_d1 = weight_d1
            w_d2 = 1 - weight_d1
            # 处理RGB图像
            rgb_attention = self.rgb_attention(input_rgb)
            rgb_features = self.rgb_feature_extractor(rgb_attention)
            fusion = w_d1 * input_rgb + w_d2 * (input_dsm + rgb_features)
        return fusion


class SmallScaleAttentionModulePlus(nn.Module):
    def __init__(self, in_channels):
        super(SmallScaleAttentionModulePlus, self).__init__()
        self.in_channels = in_channels
        reduction = 16

        # Define pooling layers
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        self.avg_pool4 = nn.AvgPool2d(kernel_size=4, stride=4, padding=1)

        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=4, stride=4, padding=1)

        # Define convolution layers for attention
        self.conv_avg2 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_avg3 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_avg4 = nn.Conv2d(in_channels, in_channels // reduction, 1)

        self.conv_max2 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_max3 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_max4 = nn.Conv2d(in_channels, in_channels // reduction, 1)

        # Convolution to combine attention maps
        self.combine_conv = nn.Conv2d(in_channels // reduction * 6, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # Apply average pooling and convolution with ReLU activation
        y_avg2 = F.interpolate(self.relu(self.conv_avg2(self.avg_pool2(x))), size=(h, w), mode='bilinear', align_corners=True)
        y_avg3 = F.interpolate(self.relu(self.conv_avg3(self.avg_pool3(x))), size=(h, w), mode='bilinear', align_corners=True)
        y_avg4 = F.interpolate(self.relu(self.conv_avg4(self.avg_pool4(x))), size=(h, w), mode='bilinear', align_corners=True)

        # Apply max pooling and convolution with ReLU activation
        y_max2 = F.interpolate(self.relu(self.conv_max2(self.max_pool2(x))), size=(h, w), mode='bilinear', align_corners=True)
        y_max3 = F.interpolate(self.relu(self.conv_max3(self.max_pool3(x))), size=(h, w), mode='bilinear', align_corners=True)
        y_max4 = F.interpolate(self.relu(self.conv_max4(self.max_pool4(x))), size=(h, w), mode='bilinear', align_corners=True)

        # Concatenate the pooled features
        y = torch.cat([y_avg2, y_avg3, y_avg4, y_max2, y_max3, y_max4], dim=1)

        # Combine attention maps
        gama = self.sigmoid(self.combine_conv(y))

        # Apply attention
        out = x * gama
        return out

# 融合后添加多尺度信息
class fusion_cwf_MS(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(fusion_cwf_MS, self).__init__()

        ################MultiScaleFeatureExtractor####################
        self.rgb_attention = SmallScaleAttentionModulePlus(in_channels)
        #####################################  CWF weight d1 prediction  ############################################
        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.SPALayer1 = SPALayer(in_channels * 2)
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input_rgb, input_dsm):
        ablation = False

        if ablation:
            fusion = torch.cat((input_dsm, input_rgb), dim=1)
            fusion = self.conv(fusion)
        else:
            d_1 = torch.cat((input_rgb, input_dsm), dim=1)
            d_1 = self.SPALayer1(d_1)
            weight_d1 = self.d1_weight_classifier(d_1)
            weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
            w_d1 = weight_d1
            w_d2 = 1 - weight_d1

            fusion = w_d1 * input_rgb + w_d2 * input_dsm

            fusion = self.rgb_attention(fusion) * self.gamma + fusion
            # fusion = self.rgb_feature_extractor(fusion)
        return fusion

# 删除自创模块前多尺度获取的注意力权重分数
class fusion_cwf_SmallScaleAttention_plus(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(fusion_cwf_SmallScaleAttention_plus, self).__init__()

        ################MultiScaleFeatureExtractor####################
        self.rgb_attention = SmallScaleAttentionModule(in_channels)
        self.rgb_feature_extractor = SmallScaleFeatureExtractor(in_channels, in_channels)
        #####################################  CWF weight d1 prediction  ############################################
        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.SPALayer1 = SPALayer(in_channels * 2)
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)

    def forward(self, input_rgb, input_dsm):
        ablation = False

        if ablation:
            fusion = torch.cat((input_dsm, input_rgb), dim=1)
            fusion = self.conv(fusion)
        else:
            d_1 = torch.cat((input_rgb, input_dsm), dim=1)
            d_1 = self.SPALayer1(d_1)
            weight_d1 = self.d1_weight_classifier(d_1)
            weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
            w_d1 = weight_d1
            w_d2 = 1 - weight_d1
            # 处理RGB图像
            rgb_attention = self.rgb_attention(input_rgb)
            rgb_features = self.rgb_feature_extractor(rgb_attention)
            fusion = w_d1 * input_rgb + w_d2 * input_dsm + rgb_features
        return fusion


# 交叉融合模块
class CrossAttention(nn.Module):
    def __init__(self, inc1, inc2):
        super(CrossAttention, self).__init__()
        self.midc1 = torch.tensor(inc1 // 4)
        self.midc2 = torch.tensor(inc2 // 4)

        self.bn_x1 = nn.BatchNorm2d(inc1)
        self.bn_x2 = nn.BatchNorm2d(inc2)

        self.kq1 = nn.Linear(inc1, self.midc2 * 2)
        self.kq2 = nn.Linear(inc2, self.midc2 * 2)

        self.v_conv = nn.Linear(inc1, 2 * self.midc1)
        self.out_conv = nn.Linear(2 * self.midc1, inc1)

        self.bn_last = nn.BatchNorm2d(inc1)
        self.dropout = nn.Dropout(0.2)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigomid = nn.Sigmoid()
        self._init_weight()

    def forward(self, fusion, input_rgb, input_dsm):
        batch_size = fusion.size(0)
        h = fusion.size(2)
        w = fusion.size(3)

        input_rgb = self.bn_x1(input_rgb)
        input_dsm = self.bn_x2(input_dsm)

        kq1 = self.kq1(input_rgb.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq2 = self.kq2(input_dsm.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq = torch.cat([kq1, kq2], dim=2)
        batchsize, N, num_channels = kq.size()
        channels_per_group = num_channels // 2
        kq = kq.view(batchsize, N, 2, channels_per_group)
        kq = kq.permute(0, 1, 3, 2).contiguous()
        kq = kq.view(batchsize, N, -1)

        k1, q1, k2, q2 = torch.split(kq, self.midc2, dim=2)

        v = self.v_conv(fusion.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        v1, v2 = torch.split(v, self.midc1, dim=2)

        mat = torch.matmul(q1, k1.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v1 = torch.matmul(mat, v1)

        mat = torch.matmul(q2, k2.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v2 = torch.matmul(mat, v2)

        v = torch.cat([v1, v2], dim=2).view(batch_size, h, w, -1)
        v = self.out_conv(v)
        v = v.permute(0, 3, 1, 2)
        v = self.bn_last(v)
        # v = self.gamma * v + fusion  # CrossAttention
        v = self.sigomid(v) * fusion * self.gamma + fusion  # CrossAttention2
        # v = self.sigomid(v) * fusion  # CrossAttention3


        return v

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

# 增强dsm支路
class fusion_cwf_DE(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(fusion_cwf_DE, self).__init__()

        #####################################  CWF weight d1 prediction  ############################################
        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.SPALayer1 = SPALayer(in_channels * 2)
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input_rgb, input_dsm):
        ablation = False

        if ablation:
            fusion = torch.cat((input_dsm, input_rgb), dim=1)
            fusion = self.conv(fusion)
        else:
            d_1 = torch.cat((input_rgb, input_dsm), dim=1)
            d_1 = self.SPALayer1(d_1)
            weight_d1 = self.d1_weight_classifier(d_1)
            weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
            w_d1 = weight_d1
            w_d2 = 1 - weight_d1
            fusion = w_d1 * input_rgb + w_d2 * input_dsm
            input_dsm = w_d2 * input_dsm * self.gamma + input_dsm

        return fusion, input_dsm



# 大师兄的模块优化
class CWGFM(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, fc_ratio = 16):
        super(CWGFM, self).__init__()

        self.d1_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d1_weight_classifier = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.SPALayer1 = SPALayer(in_channels * 2)
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)

        self.conv4 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn4 = nn.BatchNorm2d(in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(in_channels//fc_ratio, in_channels, 1, 1),
            nn.Sigmoid()
        )

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, rgb, dsm):
        u_rgb = rgb.clone()
        u_dsm = dsm.clone()

        c_attn = self.avg_pool(dsm)
        c_attn = self.fc(c_attn)
        c_attn = u_dsm * c_attn

        s_max_out, _ = torch.max(rgb, dim=1, keepdim=True)
        s_avg_out = torch.mean(rgb, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u_rgb * s_attn

        d_1 = torch.cat((rgb, dsm), dim=1)
        d_1 = self.SPALayer1(d_1)
        weight_d1 = self.d1_weight_classifier(d_1)
        weight_d1 = self.d1_weight_classifier_avgpool(weight_d1)
        w_d1 = weight_d1
        w_d2 = 1 - weight_d1
        fusion = w_d1 * c_attn + w_d2 * s_attn

        return fusion

if __name__ == '__main__':
    from thop import profile, clever_format

    input1 = torch.randn(1, 256, 64, 64)
    input2 = torch.randn(1, 256, 64, 64)
    net = CWGFM(256, 256)
    flops, params = profile(net, inputs=(input1, input2))
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPs:", macs)
    print("params:", params)
    result_dict = {}
    result_dict["模型格式"] = "fusion_cwf_MSFE"
    result_dict["FLOPs"] = macs
    result_dict["params"] = params
    # 创建一个DataFrame
    df = pd.DataFrame([result_dict])
    # 获取当前脚本（utils.py）所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(current_dir, 'FLOPs.csv')

    # 检查CSV文件是否存在
    if not os.path.exists(csv_filename):
        # 如果文件不存在，保存第一次数据并包含列名
        df.to_csv(csv_filename, index=False)
    else:
        # 如果文件存在，以追加模式打开文件并写入后续数据（不包含列名）
        with open(csv_filename, 'a', newline='') as f:
            df.to_csv(f, header=False, index=False)
