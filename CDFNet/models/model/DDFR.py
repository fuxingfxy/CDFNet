import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
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


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
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


class ConvBNx(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNx, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0)
                      ),
            norm_layer(out_channels),
        )


class ConvBNy(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNy, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2)
                      ),
            norm_layer(out_channels),
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


class GlobalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.Conv2d(dim, dim, kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.Conv2d(dim, dim, kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        # ȷ����Ⱥ͸߶��� self.ws �ı���
        if W % self.ws != 0 or H % self.ws != 0:
            W_new = ((W + self.ws - 1) // self.ws) * self.ws  # ����ȡ��������� self.ws �ı���
            H_new = ((H + self.ws - 1) // self.ws) * self.ws
            x = F.pad(x, (0, W_new - W, 0, H_new - H), mode='reflect')  # ʹ�� reflect ���

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape

        assert Wp % self.ws == 0 and Hp % self.ws == 0, "Height and Width must be divisible by window size."

        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v
        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]
        return out


class Window_basedGlobalTransformerModule(nn.Module):
    expansion = 1
    def __init__(self, dim=256, outdim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8, C=0, H=0, W=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.norm2(x)

        return x


class Channel_Selection(nn.Module):
    def __init__(self, channels, ratio=8):
        super(Channel_Selection, self).__init__()

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        self.fc_layers = nn.Sequential(
            Conv(channels, channels // ratio, kernel_size=1),
            nn.ReLU(),
            Conv(channels // ratio, channels, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c, 1, 1)
        max_x = self.max_pooling(x).view(b, c, 1, 1)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c, 1, 1)

        return v

class DynamicLocalFeatureExtractionModule(nn.Module):
    def __init__(self, dim, ratio=8, mode='v'):
        super(DynamicLocalFeatureExtractionModule, self).__init__()

        self.preconv = ConvBN(dim, dim, kernel_size=3)

        self.Channel_Selection = Channel_Selection(channels=dim, ratio=ratio)

        if mode == 'v':
            self.convbase = ConvBNx(in_channels=dim, out_channels=dim, kernel_size=3)
            self.convlarge = ConvBNx(in_channels=dim, out_channels=dim, kernel_size=5)
        else:
            self.convbase = ConvBNy(in_channels=dim, out_channels=dim, kernel_size=3)
            self.convlarge = ConvBNy(in_channels=dim, out_channels=dim, kernel_size=5)

        self.post_conv = SeparableConvBNReLU(dim, dim, 3)

    def forward(self, x):
        x = self.preconv(x)
        s = self.Channel_Selection(x)
        x = self.post_conv(s * self.convbase(x) + (1 - s) * self.convlarge(x))

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialBranch(nn.Module):
    def __init__(self, in_channels, nums_heads=16, weight_ratio=1.0, window_size=8):
        super(SpatialBranch, self).__init__()
        self.weight_ratio = weight_ratio
        self.global_v = Window_basedGlobalTransformerModule(dim=in_channels, outdim=in_channels, num_heads=nums_heads, window_size=window_size)
        self.local_v = DynamicLocalFeatureExtractionModule(in_channels, ratio=8, mode='v')
        self.conv_v = ConvBlock(in_channels=in_channels, out_channels=in_channels)

        self.global_h = Window_basedGlobalTransformerModule(dim=in_channels, outdim=in_channels, num_heads=nums_heads, window_size=window_size)
        self.local_h = DynamicLocalFeatureExtractionModule(in_channels, ratio=8, mode='h')
        self.conv_h = ConvBlock(in_channels=in_channels, out_channels=in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        vf = x.clone()
        x_v = self.global_v(vf)

        x_v = x_v + self.local_v(vf)
        x_v = self.conv_v(x_v)

        hf = x_v.clone()
        x_h = self.global_h(hf)
        x_h = x_h + self.local_h(hf)
        x_h = self.conv_h(x_h)

        return x_h


class FrequencyBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.high_attention = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.restore = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.low_process = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )


    def hi_lofi(self, x):
        B, C, H, W = x.shape

        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]

        high_feats = torch.cat([y_HL, y_LH, y_HH], dim=1)
        high_feats = self.restore(high_feats)
        high_feats = high_feats * self.high_attention(high_feats)

        low_feats = self.low_process(yL)
        fused_feats = torch.cat([high_feats, low_feats], dim=1)
        out = self.fusion(fused_feats)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

    def forward(self, x):
        return self.hi_lofi(x)


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class DDFR(nn.Module):
    def __init__(self, in_ch, out_ch, num_heads=8, window_size=8):
        super(DDFR, self).__init__()
        self.spatial = SpatialBranch(in_ch, num_heads, window_size=window_size)
        self.frequency = FrequencyBranch(dim=in_ch)
        self.conv_bn_relu_spa = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_bn_relu_fre = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, imagename=None):
        resh, resw = x.size()[-2:]
        spa = self.spatial(x)
        spa = self.conv_bn_relu_spa(spa)

        fre = self.frequency(x)
        fre = self.conv_bn_relu_fre(fre)

        return spa,fre

