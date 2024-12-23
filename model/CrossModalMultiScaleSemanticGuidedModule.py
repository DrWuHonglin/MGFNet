import datetime
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np



class PAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(PAFEM, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
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
        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))


# 分组Non-local注意力
class PGQAFEM(nn.Module):
    def __init__(self, dim, in_dim, num_groups=8):
        super(PGQAFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.num_groups = num_groups

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)

        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        group_channels = C // self.num_groups
        query2 = self.query_conv2(conv2).reshape(m_batchsize, self.num_groups, group_channels, -1).permute(0, 1, 3, 2)
        key2 = self.key_conv2(conv2).reshape(m_batchsize, self.num_groups, group_channels, -1).permute(0, 1, 3, 2)
        value2 = self.value_conv2(conv2).reshape(m_batchsize, self.num_groups, group_channels, -1).permute(0, 1, 3, 2)
        energy2 = torch.matmul(query2, key2.transpose(-2, -1))
        attention2 = self.softmax(energy2)
        out2 = torch.matmul(attention2, value2).permute(0, 1, 3, 2).reshape(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + conv2

        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        query3 = self.query_conv3(conv3).reshape(m_batchsize, self.num_groups, group_channels, -1).permute(0, 1, 3, 2)
        key3 = self.key_conv3(conv3).reshape(m_batchsize, self.num_groups, group_channels, -1).permute(0, 1, 3, 2)
        value3 = self.value_conv3(conv3).reshape(m_batchsize, self.num_groups, group_channels, -1).permute(0, 1, 3, 2)
        energy3 = torch.matmul(query3, key3.transpose(-2, -1))
        attention3 = self.softmax(energy3)
        out3 = torch.matmul(attention3, value3).permute(0, 1, 3, 2).reshape(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3

        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        query4 = self.query_conv4(conv4).reshape(m_batchsize, self.num_groups, group_channels, -1).permute(0, 1, 3, 2)
        key4 = self.key_conv4(conv4).reshape(m_batchsize, self.num_groups, group_channels, -1).permute(0, 1, 3, 2)
        value4 = self.value_conv4(conv4).reshape(m_batchsize, self.num_groups, group_channels, -1).permute(0, 1, 3, 2)
        energy4 = torch.matmul(query4, key4.transpose(-2, -1))
        attention4 = self.softmax(energy4)
        out4 = torch.matmul(attention4, value4).permute(0, 1, 3, 2).reshape(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',
                              align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))

# 增加一层空洞率2，4，6，8
class PAFEM2(nn.Module):
    def __init__(self, dim, in_dim):
        super(PAFEM2, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=8, padding=8), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv5 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv5 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv5 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma5 = nn.Parameter(torch.zeros(1))

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(6 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
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
        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        conv5 = self.conv5(x)
        m_batchsize, C, height, width = conv5.size()
        proj_query5 = self.query_conv5(conv5).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key5 = self.key_conv5(conv5).view(m_batchsize, -1, width * height)
        energy5 = torch.bmm(proj_query5, proj_key5)
        attention5 = self.softmax(energy5)
        proj_value5 = self.value_conv5(conv5).view(m_batchsize, -1, width * height)
        out5 = torch.bmm(proj_value5, attention5.permute(0, 2, 1))
        out5 = out5.view(m_batchsize, C, height, width)
        out5 = self.gamma5 * out5 + conv5

        conv6 = F.interpolate(self.conv6(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, out5, conv6), 1))


# 删除注意力机制
class PAFEMNoAttention(nn.Module):
    def __init__(self, dim, in_dim):
        super(PAFEMNoAttention, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)
        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))



class MultiModalPAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(MultiModalPAFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        self.dsm_down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
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
        ################DSM_conv2#####################
        self.dsm_conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=5, padding=5), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.dsm_query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.dsm_key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.dsm_value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.dsm_gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))
        ################DSM_conv3#####################
        self.dsm_conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=7, padding=7), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.dsm_query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.dsm_key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.dsm_value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.dsm_gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))
        ################DSM_conv4#####################
        # self.dsm_conv4 = nn.Sequential(
        #     nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=9, padding=9), nn.BatchNorm2d(down_dim), nn.PReLU())
        # self.dsm_query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        # self.dsm_key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        # self.dsm_value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        # self.dsm_gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.down_conv(x)
        # y = self.dsm_down_conv(y)
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
        ################DSM_conv2#####################
        dsm_conv2 = self.dsm_conv2(y)
        dsm_proj_query2 = self.dsm_query_conv2(dsm_conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        dsm_proj_key2 = self.dsm_key_conv2(dsm_conv2).view(m_batchsize, -1, width * height)
        dsm_energy2 = torch.bmm(dsm_proj_query2, dsm_proj_key2)
        dsm_attention2 = self.softmax(dsm_energy2)
        dsm_proj_value2 = self.dsm_value_conv2(dsm_conv2).view(m_batchsize, -1, width * height)
        dsm_out2 = torch.bmm(dsm_proj_value2, dsm_attention2.permute(0, 2, 1))
        dsm_out2 = dsm_out2.view(m_batchsize, C, height, width)
        out2 = self.dsm_gamma2 * dsm_out2 + out2


        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        ################DSM_conv3#####################
        dsm_conv3 = self.dsm_conv3(y)
        dsm_proj_query3 = self.dsm_query_conv3(dsm_conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        dsm_proj_key3 = self.dsm_key_conv3(dsm_conv3).view(m_batchsize, -1, width * height)
        dsm_energy3 = torch.bmm(dsm_proj_query3, dsm_proj_key3)
        dsm_attention3 = self.softmax(dsm_energy3)
        dsm_proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        dsm_out3 = torch.bmm(dsm_proj_value3, dsm_attention3.permute(0, 2, 1))
        dsm_out3 = dsm_out3.view(m_batchsize, C, height, width)
        out3 = self.dsm_gamma3 * dsm_out3 + out3

        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        ################DSM_conv4#####################
        # dsm_conv4 = self.dsm_conv4(y)
        # dsm_proj_query4 = self.dsm_query_conv4(dsm_conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # dsm_proj_key4 = self.dsm_key_conv4(dsm_conv4).view(m_batchsize, -1, width * height)
        # dsm_energy4 = torch.bmm(dsm_proj_query4, dsm_proj_key4)
        # dsm_attention4 = self.softmax(dsm_energy4)
        # dsm_proj_value4 = self.value_conv4(dsm_conv4).view(m_batchsize, -1, width * height)
        # dsm_out4 = torch.bmm(dsm_proj_value4, dsm_attention4.permute(0, 2, 1))
        # dsm_out4 = dsm_out4.view(m_batchsize, C, height, width)
        # out4 = self.dsm_gamma4 * dsm_conv4 + out4
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))


# 通过跨空的卷积注意力分数增强特征
class EPAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(EPAFEM, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
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



        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1)) + torch.bmm(proj_value3, attention2.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3


        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1)) + torch.bmm(proj_value4, attention3.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4


        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))



# 添加最大池化
class PAFEM_MAX(nn.Module):
    def __init__(self, dim, in_dim):
        super(PAFEM_MAX, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.gamma5 = nn.Parameter(torch.zeros(1))

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
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

        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3

        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        # Global average pooling and max pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        conv5_avg = F.interpolate(self.conv5(avg_pool), size=x.size()[2:], mode='bilinear', align_corners=True)
        conv5_max = F.interpolate(self.conv5(max_pool), size=x.size()[2:], mode='bilinear', align_corners=True)
        conv5 = conv5_avg + self.gamma5 * conv5_max


        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))




class L2Attention(nn.Module):
    def __init__(self, in_channels):
        super(L2Attention, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]

        l2_norm = torch.cdist(x_flat, x_flat, p=2)  # [B, N, N] 计算L2范数
        attention = F.softmax(-l2_norm, dim=-1)  # 负号使得距离越小权重越大

        attended_features = torch.bmm(attention, x_flat)  # [B, N, C]
        attended_features = attended_features.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]

        return attended_features

# 添加L2范数注意力
class L2PAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(L2PAFEM, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.l2_attention = L2Attention(down_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
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

        l2_attention2 = self.l2_attention(conv2)
        out2 = out2 + self.gamma2 * l2_attention2

        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3

        l2_attention3 = self.l2_attention(conv3)
        out3 = out3 + self.gamma3 * l2_attention3

        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        l2_attention4 = self.l2_attention(conv4)
        out4 = out4 + self.gamma3 * l2_attention4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))

# 多模态金字塔注意力特征增强模块
class MMFPAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(MMFPAFEM, self).__init__()
        self.down_conv_rgb = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        self.down_conv_dsm = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        self.down_conv_fusion3 = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())

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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rgb, input_dsm, fusion3):
        # x = self.down_conv_fusion3(fusion3)
        x_rgb = self.down_conv_rgb(input_rgb)
        x_dsm = self.down_conv_dsm(input_dsm)

        conv1 = self.conv1(x_rgb)
        conv2 = self.conv2(x_rgb)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + conv2
        conv3 = self.conv3(x_rgb)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        conv4 = self.conv4(x_dsm)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x_rgb, 1)), size=x_rgb.size()[2:], mode='bilinear',
                              align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))

# 添加池化针对小尺度信息
class PPAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(PPAFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.combine_conv = nn.Conv2d(down_dim * 4, down_dim, kernel_size=1, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)

        avgpool2 = self.avgpool2(conv1)
        avgpool3 = self.avgpool3(conv1)
        maxpool2 = self.maxpool2(conv1)
        maxpool3 = self.maxpool3(conv1)
        # 使用1x1卷积调整池化后的特征通道数
        avgpool2 = nn.functional.interpolate(avgpool2, size=conv1.size()[2:], mode='bilinear', align_corners=False)
        avgpool3 = nn.functional.interpolate(avgpool3, size=conv1.size()[2:], mode='bilinear', align_corners=False)
        maxpool2 = nn.functional.interpolate(maxpool2, size=conv1.size()[2:], mode='bilinear', align_corners=False)
        maxpool3 = nn.functional.interpolate(maxpool3, size=conv1.size()[2:], mode='bilinear', align_corners=False)
        # 将池化特征拼接在一起
        y = torch.cat([avgpool2, avgpool3, maxpool2, maxpool3], dim=1)
        # 通过1x1卷积层生成注意力分数
        attention = self.sigmoid(self.combine_conv(y))
        conv1 = conv1 * attention

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

        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3

        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))


# 添加池化针对小尺度信息
class PEPAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(PEPAFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.combine_conv = nn.Conv2d(down_dim * 4, down_dim, kernel_size=1, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)

        avgpool2 = self.avgpool2(conv1)
        avgpool3 = self.avgpool3(conv1)
        maxpool2 = self.maxpool2(conv1)
        maxpool3 = self.maxpool3(conv1)
        # 使用1x1卷积调整池化后的特征通道数
        avgpool2 = nn.functional.interpolate(avgpool2, size=conv1.size()[2:], mode='bilinear', align_corners=False)
        avgpool3 = nn.functional.interpolate(avgpool3, size=conv1.size()[2:], mode='bilinear', align_corners=False)
        maxpool2 = nn.functional.interpolate(maxpool2, size=conv1.size()[2:], mode='bilinear', align_corners=False)
        maxpool3 = nn.functional.interpolate(maxpool3, size=conv1.size()[2:], mode='bilinear', align_corners=False)
        # 将池化特征拼接在一起
        y = torch.cat([avgpool2, avgpool3, maxpool2, maxpool3], dim=1)
        # 通过1x1卷积层生成注意力分数
        attention = self.sigmoid(self.combine_conv(y))
        conv1 = conv1 * attention * self.gamma1 + conv1

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

        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3

        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))

# 添加交叉注意力
class CrossPAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(CrossPAFEM, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))
        ###############CrossAttention#################
        self.query_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv_dsm = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_dsm = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.gamma_dsm = nn.Parameter(torch.zeros(1))
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        ###############CrossAttention#################

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fusion, input_rgb, input_dsm):
        x = self.down_conv(fusion)

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

        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3

        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',
                              align_corners=True)

        fusion = self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))

        ###############CrossAttention#################
        m_batchsize, C, height, width = fusion.size()
        proj_query_rgb = self.query_conv_rgb(input_rgb).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key_rgb = self.key_conv_rgb(input_rgb).view(m_batchsize, -1, width * height)
        proj_query_dsm = self.query_conv_dsm(input_dsm).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key_dsm = self.key_conv_dsm(input_dsm).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(fusion).view(m_batchsize, -1, width * height)
        energy_rgb = torch.bmm(proj_query_dsm, proj_key_rgb)
        energy_dsm = torch.bmm(proj_query_rgb, proj_key_dsm)
        attention_rgb = self.softmax(energy_rgb)
        attention_dsm = self.softmax(energy_dsm)
        v_rgb = torch.bmm(proj_value, attention_rgb.permute(0, 2, 1))
        v_rgb = v_rgb.view(m_batchsize, C, height, width)
        v_dsm = torch.bmm(proj_value, attention_dsm.permute(0, 2, 1))
        v_dsm = v_dsm.view(m_batchsize, C, height, width)
        v = v_rgb * self.gamma_rgb + v_dsm * self.gamma_dsm + fusion
        ###############CrossAttention#################

        return v



# 添加池化针对小尺度信息多尺度之前
class CrossPAFEM2(nn.Module):
    def __init__(self, dim, in_dim):
        super(CrossPAFEM2, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2
        ###############CrossAttention#################
        self.query_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_rgb = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv_dsm = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_dsm = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.gamma_dsm = nn.Parameter(torch.zeros(1))
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        ###############CrossAttention#################

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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fusion, input_rgb, input_dsm):
        ###############CrossAttention#################
        m_batchsize, C, height, width = fusion.size()
        proj_query_rgb = self.query_conv_rgb(input_rgb).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key_rgb = self.key_conv_rgb(input_rgb).view(m_batchsize, -1, width * height)
        proj_query_dsm = self.query_conv_dsm(input_dsm).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key_dsm = self.key_conv_dsm(input_dsm).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(fusion).view(m_batchsize, -1, width * height)
        energy_rgb = torch.bmm(proj_query_dsm, proj_key_rgb)
        energy_dsm = torch.bmm(proj_query_rgb, proj_key_dsm)
        attention_rgb = self.softmax(energy_rgb)
        attention_dsm = self.softmax(energy_dsm)
        v_rgb = torch.bmm(proj_value, attention_rgb.permute(0, 2, 1))
        v_rgb = v_rgb.view(m_batchsize, C, height, width)
        v_dsm = torch.bmm(proj_value, attention_dsm.permute(0, 2, 1))
        v_dsm = v_dsm.view(m_batchsize, C, height, width)
        fusion = v_rgb * self.gamma_rgb + v_dsm * self.gamma_dsm + fusion
        ###############CrossAttention#################
        x = self.down_conv(fusion)
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


        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3


        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        fusion = self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))

        return fusion

# 激励挤压金字塔
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


class SqueezeAndExcitationMaxAvg(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitationMaxAvg, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1), activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Global Average Pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        # Global Max Pooling
        max_pool = F.adaptive_max_pool2d(x, 1)

        # Squeeze and Excitation
        avg_weight = self.fc(avg_pool)
        max_weight = self.fc(max_pool)

        # Combine both weights
        weighting = avg_weight + max_weight

        # Apply weighting to the input features
        y = x * weighting
        return y

class SEPAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(SEPAFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.se_layer2 = SqueezeAndExcitationMaxAvg(down_dim)
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma_se2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.se_layer3 = SqueezeAndExcitationMaxAvg(down_dim)
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma_se3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.se_layer4 = SqueezeAndExcitationMaxAvg(down_dim)
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))
        self.gamma_se4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)

        conv2 = self.conv2(x)
        se_conv2 = self.se_layer2(conv2)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + self.gamma_se2 * se_conv2 + conv2

        conv3 = self.conv3(x)
        se_conv3 = self.se_layer3(conv3)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + self.gamma_se3 * se_conv3 + conv3

        conv4 = self.conv4(x)
        se_conv4 = self.se_layer4(conv4)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + self.gamma_se4 * se_conv4 + conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))


# 多重注意力Non-local
class MultiAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.query_conv_se = SqueezeAndExcitation(in_channels // 8)
        self.key_conv_se = SqueezeAndExcitation(in_channels // 8)
        self.value_conv_se = SqueezeAndExcitation(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):


        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv_se(self.query_conv(x)).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv_se(self.key_conv(x)).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv_se(self.value_conv(x)).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


# 跨模态语义引导
class CSGPAFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(CSGPAFEM, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        # Cross semantic Guide Model
        self.num_groups = 8
        self.query_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.query_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_dsm = nn.Parameter(torch.zeros(1))
        # Cross semantic Guide Model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fusion, input_rgb, input_dsm):
        x = self.down_conv(fusion)
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
        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        fusion = self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))
        # Cross semantic Guide Model
        batch_size, C_rgb, H, W = input_dsm.size()
        group_channels = C_rgb // self.num_groups
        query_dsm = self.query_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        key_dsm = self.key_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)

        query_rgb = self.query_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        key_rgb = self.key_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)

        value = self.value_conv(fusion).reshape(batch_size, self.num_groups, group_channels, -1)

        # Cross-attention
        query_dsm = query_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_dsm = key_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        query_rgb = query_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_rgb = key_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        attn_dsm = torch.matmul(query_dsm, key_dsm.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_dsm = self.softmax(attn_dsm)
        out_dsm = torch.matmul(attn_dsm, value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_dsm = out_dsm.reshape(batch_size, C_rgb, H, W)
        attn_rgb = torch.matmul(query_rgb, key_rgb.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_rgb = self.softmax(attn_rgb)
        out_rgb = torch.matmul(attn_rgb, value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_rgb = out_rgb.reshape(batch_size, C_rgb, H, W)

        fusion = out_dsm * self.gamma_dsm + fusion + out_rgb * self.gamma_rgb
        # Cross semantic Guide Model
        return fusion

# 跨模态多尺度语义引导模块
class CMSGM(nn.Module):
    def __init__(self, dim, in_dim):
        super(CMSGM, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        # Cross semantic Guide Model
        self.num_groups = 8
        self.query_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.query_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_dsm = nn.Parameter(torch.zeros(1))
        # Cross semantic Guide Model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fusion, input_rgb, input_dsm):
        x = self.down_conv(fusion)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)

        energy2 = torch.bmm(proj_query3, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + conv2

        energy3 = torch.bmm(proj_query4, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3

        energy4 = torch.bmm(proj_query2, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        fusion = self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))
        # Cross semantic Guide Model
        batch_size, C_rgb, H, W = input_dsm.size()
        group_channels = C_rgb // self.num_groups
        query_dsm = self.query_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        key_dsm = self.key_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)

        query_rgb = self.query_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        key_rgb = self.key_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)

        value = self.value_conv(fusion).reshape(batch_size, self.num_groups, group_channels, -1)

        # Cross-attention
        query_dsm = query_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_dsm = key_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        query_rgb = query_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_rgb = key_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        attn_dsm = torch.matmul(query_dsm, key_dsm.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_dsm = self.softmax(attn_dsm)
        out_dsm = torch.matmul(attn_dsm, value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_dsm = out_dsm.reshape(batch_size, C_rgb, H, W)
        attn_rgb = torch.matmul(query_rgb, key_rgb.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_rgb = self.softmax(attn_rgb)
        out_rgb = torch.matmul(attn_rgb, value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_rgb = out_rgb.reshape(batch_size, C_rgb, H, W)

        fusion = out_dsm * self.gamma_dsm + fusion + out_rgb * self.gamma_rgb
        # Cross semantic Guide Model
        return fusion



# 跨模态语义感知金字塔注意力模块
class CMSGMNoMIM(nn.Module):
    def __init__(self, dim, in_dim):
        super(CMSGMNoMIM, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        # Cross semantic Guide Model
        self.num_groups = 8
        self.query_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.query_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_dsm = nn.Parameter(torch.zeros(1))
        # Cross semantic Guide Model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fusion, input_rgb, input_dsm):
        x = self.down_conv(fusion)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        out2 = conv2
        conv3 = self.conv3(x)
        out3 = conv3
        conv4 = self.conv4(x)
        out4 = conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)
        fusion = self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))
        # Cross semantic Guide Model
        batch_size, C_rgb, H, W = input_dsm.size()
        group_channels = C_rgb // self.num_groups
        query_dsm = self.query_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        key_dsm = self.key_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)

        query_rgb = self.query_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        key_rgb = self.key_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)

        value = self.value_conv(fusion).reshape(batch_size, self.num_groups, group_channels, -1)

        # Cross-attention
        query_dsm = query_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_dsm = key_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        query_rgb = query_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_rgb = key_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        attn_dsm = torch.matmul(query_dsm, key_dsm.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_dsm = self.softmax(attn_dsm)
        out_dsm = torch.matmul(attn_dsm, value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_dsm = out_dsm.reshape(batch_size, C_rgb, H, W)
        attn_rgb = torch.matmul(query_rgb, key_rgb.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_rgb = self.softmax(attn_rgb)
        out_rgb = torch.matmul(attn_rgb, value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_rgb = out_rgb.reshape(batch_size, C_rgb, H, W)

        fusion = out_dsm * self.gamma_dsm + fusion + out_rgb * self.gamma_rgb
        # Cross semantic Guide Model
        return fusion

# 多尺度交叉注意力模块
class MSCAM(nn.Module):
    def __init__(self, dim, in_dim):
        super(MSCAM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2

        self.in_channels = down_dim
        self.num_groups = 8
        self.group_channels = down_dim // self.num_groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 16, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(in_dim // 16, in_dim, 1, 1),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, fusion):
        x = self.down_conv(fusion)
        # conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        out2 = conv2
        conv3 = self.conv3(x)
        out3 = conv3
        conv4 = self.conv4(x)
        out4 = conv4

        batch_size, channels, height, width = conv2.size()

        # 函数，用于计算自注意力
        def self_attention(V, K, Q):
            V_reshaped = V.reshape(batch_size, self.num_groups, self.group_channels, -1)
            K_reshaped = K.reshape(batch_size, self.num_groups, self.group_channels, -1)
            Q_reshaped = Q.reshape(batch_size, self.num_groups, self.group_channels, -1)

            # 自注意力机制
            attention = torch.einsum('bgci,bgcj->bgij', Q_reshaped, K_reshaped) / (self.group_channels ** 0.5)
            attention = F.softmax(attention, dim=-1)

            # 使用注意力权重对V进行加权
            out = torch.einsum('bgij,bgcj->bgci', attention, V_reshaped)

            # reshape回原始形状
            out = out.reshape(batch_size, channels, height, width)
            return out

        # 三个不同的自注意力组合
        out_conv2 = self_attention(conv2, conv3, conv4)  # conv2 作 V, conv3 作 K, conv4 作 Q
        out_conv3 = self_attention(conv3, conv2, conv4)  # conv3 作 V, conv2 作 K, conv4 作 Q
        out_conv4 = self_attention(conv4, conv2, conv3)  # conv4 作 V, conv2 作 K, conv3 作 Q

        # 将三个输出加权求和或拼接
        out = out_conv2 + out_conv3 + out_conv4  # 这里简单做加和，可以根据需求调整
        out = self.fuse(out)

        c_attn = self.avg_pool(fusion)
        c_attn = self.fc(c_attn)
        c_attn = fusion * c_attn

        fusion = out + c_attn

        return fusion

# 多尺度模块
class MSCAM(nn.Module):
    def __init__(self, dim, in_dim):
        super(MSCAM, self).__init__()
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
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fusion):
        x = self.down_conv(fusion)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)

        energy2 = torch.bmm(proj_query3, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + conv2

        energy3 = torch.bmm(proj_query4, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3

        energy4 = torch.bmm(proj_query2, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4

        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True)

        fusion = self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))


        return fusion



if __name__ == '__main__':
    from thop import profile, clever_format

    input1 = torch.randn(1, 256, 8, 8)
    input2 = torch.randn(10, 1024, 16, 16)
    input3 = torch.randn(10, 1024, 16, 16)
    net = MSCAM(dim=256, in_dim=256)
    flops, params = profile(net, inputs=(input1,))
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPs:", macs)
    print("params:", params)
    result_dict = {}
    result_dict["时间"] = datetime.datetime.now()
    result_dict["模型格式"] = "CSGPAFEM"
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
