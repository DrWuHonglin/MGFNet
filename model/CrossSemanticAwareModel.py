import torch
import torch.nn as nn
import torch.nn.functional as F

# 最初版本
class CrossSemanticAttentionModule0(nn.Module):
    def __init__(self, in_dim, dilation=1, padding=0):
        super(CrossSemanticAttentionModule0, self).__init__()
        down_dim = in_dim // 2
        self.conv_rgb = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=dilation, padding=padding), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_rgb = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_rgb = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_rgb = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.up_rgb = nn.Conv2d(in_channels=down_dim, out_channels=in_dim, kernel_size=1)

        self.gamma_rgb = nn.Parameter(torch.zeros(1))

        self.conv_dsm = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=dilation, padding=padding), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_dsm = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_dsm = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_dsm = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.up_dsm = nn.Conv2d(in_channels=down_dim, out_channels=in_dim, kernel_size=1)
        self.gamma_dsm = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rgb, input_dsm):
        conv_rgb = self.conv_rgb(input_rgb)
        m_batchsize, C, height, width = conv_rgb.size()
        proj_query_rgb = self.query_rgb(conv_rgb).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key_rgb = self.key_rgb(conv_rgb).view(m_batchsize, -1, width * height)
        conv_dsm = self.conv_dsm(input_dsm)
        proj_query_dsm = self.query_dsm(conv_dsm).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key_dsm = self.key_dsm(conv_dsm).view(m_batchsize, -1, width * height)

        energy_rgb = torch.bmm(proj_query_dsm, proj_key_rgb)
        attention_rgb = self.softmax(energy_rgb)
        proj_value_rgb = self.value_rgb(conv_rgb).view(m_batchsize, -1, width * height)
        out_rgb = torch.bmm(proj_value_rgb, attention_rgb.permute(0, 2, 1))
        out_rgb = out_rgb.view(m_batchsize, C, height, width)
        out_rgb = self.gamma_rgb * out_rgb + conv_rgb

        energy_dsm = torch.bmm(proj_query_rgb, proj_key_dsm)
        attention_dsm = self.softmax(energy_dsm)
        proj_value_dsm = self.value_dsm(conv_dsm).view(m_batchsize, -1, width * height)
        out_dsm = torch.bmm(proj_value_dsm, attention_dsm.permute(0, 2, 1))
        out_dsm = out_dsm.view(m_batchsize, C, height, width)
        out_dsm = self.gamma_dsm * out_dsm + conv_dsm

        out_rgb = self.up_rgb(out_rgb) + input_rgb
        out_dsm = self.up_dsm(out_dsm) + input_dsm
        return out_rgb, out_dsm


# 最优版本
class CrossSemanticAttentionModule(nn.Module):
    def __init__(self, in_dim, num_groups=8):
        super(CrossSemanticAttentionModule, self).__init__()
        self.num_groups = num_groups
        self.group_channels = in_dim // num_groups

        self.query_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.query_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_dsm = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rgb, input_dsm):
        batch_size, C_rgb, H, W = input_rgb.size()
        group_channels = C_rgb // self.num_groups

        # RGB branch
        query_rgb = self.query_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        key_rgb = self.key_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        value_rgb = self.value_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)

        # DSM branch
        query_dsm = self.query_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        key_dsm = self.key_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        value_dsm = self.value_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)

        # Cross-attention
        query_rgb = query_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_dsm = key_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        attn_rgb_to_dsm = torch.matmul(query_rgb, key_dsm.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_rgb_to_dsm = self.softmax(attn_rgb_to_dsm)

        out_dsm = torch.matmul(attn_rgb_to_dsm, value_dsm.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_dsm = out_dsm.reshape(batch_size, C_rgb, H, W)

        query_dsm = query_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]
        key_rgb = key_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]

        attn_dsm_to_rgb = torch.matmul(query_dsm, key_rgb.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_dsm_to_rgb = self.softmax(attn_dsm_to_rgb)

        out_rgb = torch.matmul(attn_dsm_to_rgb, value_rgb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_rgb = out_rgb.reshape(batch_size, C_rgb, H, W)

        # Apply residual connection
        out_rgb = self.gamma_rgb * out_rgb + input_rgb
        out_dsm = self.gamma_dsm * out_dsm + input_dsm

        return out_rgb, out_dsm


# 语义感知模块 原始特征残差 + 共享权重提取特征残差
class CrossSemanticAttentionModule2(nn.Module):
    def __init__(self, in_dim=1024, num_groups=8, dilation=2, padding=2):
        super(CrossSemanticAttentionModule2, self).__init__()
        self.num_groups = num_groups
        down_dim = in_dim // 2
        self.group_channels = down_dim // num_groups

        # Downsample convolution layer for RGB and DSM with shared weights
        self.shared_down_conv = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=dilation, padding=padding),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        # Attention mechanisms for RGB and DSM
        self.query_conv_rgb = nn.Conv2d(down_dim, down_dim, kernel_size=1)
        self.key_conv_rgb = nn.Conv2d(down_dim, down_dim, kernel_size=1)
        self.value_conv_rgb = nn.Conv2d(down_dim, down_dim, kernel_size=1)

        self.query_conv_dsm = nn.Conv2d(down_dim, down_dim, kernel_size=1)
        self.key_conv_dsm = nn.Conv2d(down_dim, down_dim, kernel_size=1)
        self.value_conv_dsm = nn.Conv2d(down_dim, down_dim, kernel_size=1)

        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_dsm = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        # Upsample convolution layer to restore original dimensions
        self.shared_up_conv = nn.Conv2d(down_dim, in_dim, kernel_size=1)

    def forward(self, input_rgb, input_dsm):
        batch_size, C_rgb, H, W = input_rgb.size()
        group_channels = self.group_channels

        # Downsample input channels
        down_input_rgb = self.shared_down_conv(input_rgb)
        down_input_dsm = self.shared_down_conv(input_dsm)

        # RGB branch
        query_rgb = self.query_conv_rgb(down_input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        key_rgb = self.key_conv_rgb(down_input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)
        value_rgb = self.value_conv_rgb(down_input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)

        # DSM branch
        query_dsm = self.query_conv_dsm(down_input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        key_dsm = self.key_conv_dsm(down_input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        value_dsm = self.value_conv_dsm(down_input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)

        # Cross-attention
        query_rgb = query_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_dsm = key_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        attn_rgb_to_dsm = torch.matmul(query_rgb, key_dsm.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_rgb_to_dsm = self.softmax(attn_rgb_to_dsm)

        out_dsm = torch.matmul(attn_rgb_to_dsm, value_dsm.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_dsm = out_dsm.reshape(batch_size, group_channels * self.num_groups, H, W)

        query_dsm = query_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]
        key_rgb = key_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]

        attn_dsm_to_rgb = torch.matmul(query_dsm, key_rgb.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_dsm_to_rgb = self.softmax(attn_dsm_to_rgb)

        out_rgb = torch.matmul(attn_dsm_to_rgb, value_rgb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_rgb = out_rgb.reshape(batch_size, group_channels * self.num_groups, H, W)

        # Apply residual connection
        out_rgb = self.gamma_rgb * out_rgb + down_input_rgb
        out_dsm = self.gamma_dsm * out_dsm + down_input_dsm

        # Upsample output channels
        out_rgb = self.shared_up_conv(out_rgb) + input_rgb
        out_dsm = self.shared_up_conv(out_dsm) + input_dsm

        return out_rgb, out_dsm


class CAGFM(nn.Module):
    def __init__(self, in_dim, num_groups=8):
        super(CAGFM, self).__init__()
        self.num_groups = num_groups
        self.group_channels = in_dim // num_groups

        self.query_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv_rgb = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.key_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv_dsm = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_dsm = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rgb, input_dsm):
        batch_size, C_rgb, H, W = input_rgb.size()
        group_channels = C_rgb // self.num_groups

        # RGB branch
        query_rgb = self.query_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)

        value_rgb = self.value_conv_rgb(input_rgb).reshape(batch_size, self.num_groups, group_channels, -1)

        # DSM branch
        key_dsm = self.key_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)
        value_dsm = self.value_conv_dsm(input_dsm).reshape(batch_size, self.num_groups, group_channels, -1)

        # Cross-attention
        query_rgb = query_rgb.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_rgb]
        key_dsm = key_dsm.permute(0, 1, 3, 2)  # [batch_size, num_groups, H*W, group_channels_dsm]

        attn_rgb_to_dsm = torch.matmul(query_rgb, key_dsm.transpose(-2, -1))  # [batch_size, num_groups, H*W, H*W]
        attn_rgb_to_dsm = self.softmax(attn_rgb_to_dsm)

        out_dsm = torch.matmul(attn_rgb_to_dsm, value_dsm.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_dsm = out_dsm.reshape(batch_size, C_rgb, H, W)



        out_rgb = torch.matmul(1 - attn_rgb_to_dsm, value_rgb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out_rgb = out_rgb.reshape(batch_size, C_rgb, H, W)


        return out_rgb + out_dsm



if __name__ == '__main__':
    from thop import profile, clever_format

    input1 = torch.randn(1, 128, 128, 128)
    input2 = torch.randn(1, 128, 128, 128)
    net = CAGFM(128)
    flops, params = profile(net, inputs=(input1, input2))
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPs:", macs)
    print("params:", params)
