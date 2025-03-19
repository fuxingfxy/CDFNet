import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from .DDFR import DDFR
from .CDFM import CDFM

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ChannelAtt, self).__init__()
        self.conv_bn_relu = ConvBNReLU(in_channels, out_channels, kernel_size, stride=stride)
        self.conv_1x1 = ConvBN(out_channels, out_channels, 1, stride=1)

    def forward(self, x, fre=False):
        feat = self.conv_bn_relu(x)
        if fre:
            h_tv = torch.diff(feat, dim=2).pow(2)
            w_tv = torch.diff(feat, dim=3).pow(2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, torch.sigmoid(atten)


class CSIM(nn.Module):
    def __init__(self, channels, ext=2, r=16):
        super(CSIM, self).__init__()
        self.r = r
        scale = max(1, (r * r) // channels)
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        self.spatial_mlp = nn.Sequential(nn.Linear(channels * scale, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.spatial_att = ChannelAtt(channels * ext, channels)
        self.context_mlp = nn.Sequential(nn.Linear(channels * scale, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.context_att = ChannelAtt(channels, channels)
        self.context_head = ConvBNReLU(channels, channels, 3, stride=1)
        self.smooth = ConvBN(channels, channels, 3, stride=1)

    def forward(self, sp_feat, co_feat):
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        b, c, h, w = s_att.size()
        s_att_split = F.normalize(s_att.view(b, self.r, c // self.r), p=2, dim=2)
        c_att_split = F.normalize(c_att.view(b, self.r, c // self.r), p=2, dim=2)
        chl_affinity = torch.bmm(s_att_split, c_att_split.permute(0, 2, 1))
        chl_affinity = chl_affinity.view(b, -1)
        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity))
        co_mlp_out = F.relu(self.context_mlp(chl_affinity))
        re_s_att = torch.sigmoid(s_att + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_c_att = torch.sigmoid(c_att + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1))
        c_feat = torch.mul(c_feat, re_c_att)
        s_feat = torch.mul(s_feat, re_s_att)
        c_feat = F.interpolate(c_feat, s_feat.size()[2:], mode='bilinear', align_corners=False)
        c_feat = self.context_head(c_feat)
        out = self.smooth(s_feat + c_feat)
        return s_feat, c_feat, out

class CSIM1(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super(CSIM1, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32) * 0.5 + torch.randn(2) * 0.01, requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = ConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)
        return x

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=128,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv1 = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)

        self.d3 = DDFR(in_ch=decode_channels, out_ch=decode_channels,num_heads=8,window_size=window_size)
        self.p3 = CSIM(decode_channels, ext=4)
        self.b3 = CDFM(dim=decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.d2 = DDFR(in_ch=decode_channels, out_ch=decode_channels,num_heads=8,window_size=window_size)
        self.p2 = CSIM(decode_channels, ext=2)
        self.b2 = CDFM(dim=decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.d1 = DDFR(in_ch=decode_channels, out_ch=decode_channels,num_heads=8,window_size=window_size)
        self.b1 = CDFM(dim=decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.p1 = CSIM1(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1)
        )
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        res4 = self.pre_conv1(res4)
        spa, fre = self.d3(res4)
        f1 = self.b3(spa,fre)
        x = f1 * res4 + res4

        _, _, y2 = self.p3(res3, x)
        spa1, fre1 = self.d2(y2)
        f2 = self.b2(spa1, fre1)
        x = f2 * y2 + y2

        _, _, y3 = self.p2(res2, x)
        spa2, fre2 = self.d1(y3)
        f3 = self.b1(spa2, fre2)
        x = f3 * y3 + y3

        x = self.p1(x, res1)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = self.segmentation_head(x)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class CDFNet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnext50_32x4d',
                 pretrained=True,
                 num_classes=6
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        decoder_input_channels = [64, 128, 256, 512]

        self.res1_dim = Conv(encoder_channels[0], decoder_input_channels[0], kernel_size=1)
        self.res2_dim = Conv(encoder_channels[1], decoder_input_channels[1], kernel_size=1)
        self.res3_dim = Conv(encoder_channels[2], decoder_input_channels[2], kernel_size=1)
        self.res4_dim = Conv(encoder_channels[3], decoder_input_channels[3], kernel_size=1)

        self.decoder = Decoder(decoder_input_channels, decode_channels, dropout, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]

        res1, res2, res3, res4 = self.backbone(x)

        res1 = self.res1_dim(res1)
        res2 = self.res2_dim(res2)
        res3 = self.res3_dim(res3)
        res4 = self.res4_dim(res4)
        x = self.decoder(res1, res2, res3, res4, h, w)
        return x
