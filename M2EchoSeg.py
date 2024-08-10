import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, split_args
import torch.nn.functional as F
import numpy as np

from einops import rearrange


def get_acti_layer(act, nchan: int = 0):
    if act == "prelu":
        act = ("prelu", {"num_parameters": nchan})
    act_name, act_args = split_args(act)
    act_type = Act[act_name]
    return act_type(**act_args)


class LUConv(nn.Module):
    def __init__(self, spatial_dims: int, nchan: int, act, bias: bool = False):
        super().__init__()

        self.act_function = get_acti_layer(act, nchan)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.act_function(out)
        return out


def _make_nconv(spatial_dims: int, nchan: int, depth: int, act, bias: bool = False):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(spatial_dims, nchan, act, bias))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, act, bias: bool = False
    ):
        super().__init__()

        if out_channels % in_channels != 0:
            raise ValueError(
                f"out channels should be divisible by in_channels. Got in_channels={in_channels}, out_channels={out_channels}."
            )

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_function = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        repeat_num = self.out_channels // self.in_channels
        x16 = x.repeat([1, repeat_num, 1, 1, 1][: self.spatial_dims + 2])
        out = self.act_function(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        nconvs: int,
        act,
        dropout_prob=None,
        dropout_dim: int = 3,
        bias: bool = False,
    ):
        super().__init__()

        conv_type: type[nn.Conv2d] = Conv[Conv.CONV, spatial_dims]
        norm_type: type[nn.BatchNorm2d] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: type[nn.Dropout2d] = Dropout[Dropout.DROPOUT, dropout_dim]

        out_channels = 2 * in_channels
        self.down_conv = conv_type(
            in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        self.bn1 = norm_type(out_channels)
        self.act_function1 = get_acti_layer(act, out_channels)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act, bias)
        self.dropout = dropout_type(
            dropout_prob) if dropout_prob is not None else None
        # self.dwse = DWSE(out_channels)
        # self.frm = FeatureRectifyModule(out_channels)

    def forward(self, x):
        down = self.act_function1(self.bn1(self.down_conv(x)))
        if self.dropout is not None:
            out = self.dropout(down)
        else:
            out = down
        out = self.ops(out)
        out = self.act_function2(torch.add(out, down))
        # out = self.frm(out)
        return out


class UpTransition(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        nconvs: int,
        act,
        dropout_prob=None,
        dropout_dim: int = 3,
    ):
        super().__init__()

        conv_trans_type: type[nn.ConvTranspose2d] = Conv[Conv.CONVTRANS, spatial_dims]
        norm_type: type[nn.BatchNorm2d] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: type[nn.Dropout] = Dropout[Dropout.DROPOUT, dropout_dim]
        self.up_conv = conv_trans_type(
            in_channels, out_channels // 2, kernel_size=2, stride=2)
        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn1 = norm_type(out_channels // 2)
        self.dropout = dropout_type(
            dropout_prob) if dropout_prob is not None else None
        self.dropout2 = dropout_type(0.5)
        self.act_function1 = get_acti_layer(act, out_channels // 2)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act)

    def forward(self, x, skipx):
        if self.dropout is not None:
            out = self.dropout(x)
        else:
            out = x
        skipxdo = self.dropout2(skipx)
        out = self.act_function1(self.bn1(self.up_conv(out)))
        # out = self.act_function1(self.bn1(self.up_conv(self.up(out))))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.act_function2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, act, bias: bool = False
    ):
        super().__init__()

        conv_type: type[nn.Conv2d] = Conv[Conv.CONV, spatial_dims]

        self.act_function1 = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )
        self.conv2 = conv_type(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv_block(x)
        out = self.act_function1(out)
        out = self.conv2(out)
        return out


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.sc = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size, stride,
                      padding, dilation, groups=inplanes, bias=bias),
            nn.BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        )

    def forward(self, x):
        x = self.sc(x)
        return x


class OnePlusTanh(nn.Module):
    def __init__(self,):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x) + 1


class MSCAM_a(nn.Module):
    def __init__(self, c_in, c_feat, c_atten, T):
        super(MSCAM_a, self).__init__()
        self.c_feat = c_feat
        self.c_atten = c_atten
        self.T = T
        self.conv_feat = nn.Conv2d(c_in, c_feat, kernel_size=1)
        self.conv_atten = nn.Conv2d(c_in, c_atten, kernel_size=1)

    def forward(self, input: torch.Tensor):
        bt, c, h, w = input.size()
        b = bt // self.T
        feat = self.conv_feat(rearrange(input, '(b t) c h w -> b t c h w',
                              b=b, t=self.T)[:, 0]).view(b, self.c_feat, -1)  # b c hw
        atten = rearrange(self.conv_atten(
            input), '(b t) N h w -> b (t N) (h w)', b=b, t=self.T, N=self.c_atten)  # b TN hw
        # TODO: softmax
        atten = F.softmax(atten, dim=-1)
        context = torch.bmm(feat, atten.permute(0, 2, 1))  # b c TN

        return context, atten


class MSCAM_b(nn.Module):
    def __init__(self, c_atten, c_input, T):
        super(MSCAM_b, self).__init__()
        self.c_atten = c_atten
        self.T = T
        self.conv_input = nn.Conv2d(c_input, T*c_atten, kernel_size=1)
        # self.V_conv = nn.Conv2d(c_input, c_input, kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_input * 2, c_input, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_input),
            nn.ReLU(inplace=True),
        )

    def forward(self, context: torch.Tensor, input: torch.Tensor):
        # context: b, C, TN
        # input: b, C, H, W
        b, c, h, w = input.size()
        P = self.conv_input(input)  # b, TN, H, W
        # softmax1
        # att_map = context.matmul(F.softmax(rearrange(P, 'b N h w -> b N (h w)'), dim=-1))  # b C HW
        # softmax2
        att_map = F.softmax(context.matmul(
            rearrange(P, 'b N h w -> b N (h w)')), dim=-1)  # b C HW

        #  V = rearrange(self.V_conv(input), 'b c h w -> b c (h w)', h=h, w=w)  # b C (H W)
        # new_V = F.softmax(rearrange(context, 'b C N -> b N C').matmul(V), dim=-1) # B, TN, HW
        # new_P = rearrange(context.matmul(new_V), 'b c (h w) -> b c h w', h=h, w=w) # B, C, H, W
        att_map = rearrange(att_map, 'b C (H W) -> b C H W', H=h, W=w)
        # output = self.out_conv(torch.cat([input, att_map], dim=1))
        # output = self.out_conv(input + att_map)
        output = self.out_conv(torch.cat([input, att_map], dim=1))
        return output, att_map


class MVFRM(nn.Module):
    def __init__(self, c_a, c_b, c_att):
        super(MVFRM, self).__init__()
        self.c_a = c_a
        self.c_b = c_b
        # self.gate_conv = nn.Conv2d(c_a, c_att, kernel_size=1)
        self.channel_conv = nn.Conv1d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor, gate_map):
        b, c, h, w = input_a.size()
        # plot_3d_mat(F.sigmoid(input_de.transpose(-1, -2)), j, '.', name=f'input_de_{i}')
        # plot_3d_mat(F.sigmoid(input_en.transpose(-1, -2)), j, '.', name=f'input_en_{i}')
        input_a = input_a.view(b, self.c_a, -1)

        # Channel Resampling
        energy = input_b.view(
            b, self.c_b, -1).matmul(input_a.transpose(-1, -2))
        # Prevent loss divergence during training
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        channel_attention_map = torch.softmax(energy_new, dim=-1)
        cr = channel_attention_map.matmul(input_a).view(
            b, -1, h, w)  # channel_attention_feat
        # plot_3d_mat(F.sigmoid(input_en[0].transpose(-1, -2)), j, '.', name=f'cr_f_{i}')

        # Spatial Gating
        gate_map = torch.sigmoid(gate_map)
        sg = input_a.view(b, self.c_a, h, w).mul(gate_map)

        x = torch.mean(torch.mean(sg, dim=-1), dim=-1).view(b, -1)
        f = torch.mean(torch.mean(cr, dim=-1), dim=-1).view(b, -1)  # [B, C]
        x_f = torch.stack([x, f], dim=1)  # [B, 2, C]
        x_f_w = self.sigmoid(self.channel_conv(x_f)).squeeze(
            1).view(b, c, 1, 1)  # [B, C, 1, 1]
        f_w = 1 - x_f_w
        output = x_f_w * sg + f_w * cr
        # # plot_3d_mat(F.sigmoid(input_en[0].transpose(-1, -2)), j, '.', name=f'sg_f_{i}')

        return output
        # return cr


class M2EchoSeg(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        act=("elu", {"inplace": True}),
        dropout_prob: float = 0.5,
        dropout_dim: int = 3,
        features: int = [16, 32, 64, 128, 256, 16],
        bias: bool = False,
        n_views=2,
        video_length=72,
        N=64
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise AssertionError("spatial_dims can only be 2 or 3.")

        self.in_tr = InputTransition(
            spatial_dims, in_channels, features[0], act, bias=bias)
        self.down_tr32 = DownTransition(
            spatial_dims,  features[0], 1, act, bias=bias)
        self.down_tr64 = DownTransition(
            spatial_dims,  features[1], 2, act, bias=bias)
        self.down_tr128 = DownTransition(
            spatial_dims,  features[2], 3, act, dropout_prob=dropout_prob, bias=bias)
        self.down_tr256 = DownTransition(
            spatial_dims,  features[3], 2, act, dropout_prob=dropout_prob, bias=bias)
        self.up_tr256 = UpTransition(
            spatial_dims,  features[4],  features[4], 2, act, dropout_prob=dropout_prob)
        self.up_tr128 = UpTransition(
            spatial_dims,  features[4],  features[3], 2, act, dropout_prob=dropout_prob)
        self.up_tr64 = UpTransition(
            spatial_dims,  features[3],  features[2], 1, act)
        self.up_tr32 = UpTransition(
            spatial_dims,  features[2],  features[1], 1, act)
        self.out_tr = OutputTransition(
            spatial_dims,  features[1], out_channels, act, bias=bias)

        fea = features
        self.n_views = n_views
        self.video_length = video_length

        self.shared_fuse1 = nn.Sequential(
            nn.Conv2d(fea[2]*5, fea[2], 3, 1, 1),
            nn.BatchNorm2d(fea[2]),
            nn.ReLU(inplace=True),
        )
        self.shared_fuse2 = nn.Sequential(
            nn.Conv2d(fea[2], fea[2], 1),
            nn.BatchNorm2d(fea[2]),
            nn.ReLU(inplace=True),
        )

        self.mscam_a = nn.ModuleList(
            MSCAM_a(fea[2], fea[2], N, self.video_length) for _ in range(n_views))
        self.mscam_b = nn.ModuleList(
            MSCAM_b(N, fea[2], self.video_length) for _ in range(n_views))
        self.mvfrm = nn.ModuleList(
            MVFRM(fea[2], fea[2], N) for _ in range(n_views))
        self.gamma = nn.Parameter(torch.eye(self.n_views), requires_grad=True)

        self.fuse0 = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Conv2d(features[0], features[2], 3, 1, 1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(features[1], features[2], 3, 1, 1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(features[2], features[2], 3, 1, 1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )
        self.fuse3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(features[3], features[2], 3, 1, 1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )
        self.fuse4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv2d(features[4], features[2], 3, 1, 1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )

        self.defuse0 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv2d(fea[2], features[0], 1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )

        self.defuse1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(fea[2], features[1], 1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
        )
        self.defuse2 = nn.Sequential(
            nn.Conv2d(fea[2], features[2], 1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )
        self.defuse3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(fea[2], features[3], 1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
        )
        self.defuse4 = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Conv2d(fea[2], features[4], 1),
            nn.BatchNorm2d(features[4]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x0 = self.in_tr(x)
        x1 = self.down_tr32(x0)
        x2 = self.down_tr64(x1)
        x3 = self.down_tr128(x2)
        x4 = self.down_tr256(x3)

        lx0 = self.fuse0(x0)
        lx1 = self.fuse1(x1)
        lx2 = self.fuse2(x2)
        lx3 = self.fuse3(x3)
        lx4 = self.fuse4(x4)

        concat_x = torch.cat([lx0, lx1, lx2, lx3, lx4], dim=1)
        f = concat_x

        f = self.shared_fuse1(f)
        f = self.shared_fuse2(f)

        f_list = rearrange(f, '(b v t) c h w -> v (b t) c h w',
                           v=self.n_views, t=self.video_length)

        re_f_list = []
        cur_f_list = []
        attmap_list = []
        atten_list = []
        for i in range(self.n_views):
            context, atten = self.mscam_a[i](f_list[i])
            f_input = rearrange(
                f_list[i], '(b t) c h w -> b t c h w', t=self.video_length)[:, 0]
            f_cur, attmap = self.mscam_b[i](context, f_input)

            cur_f_list.append(f_cur)
            atten_list.append(atten)
            attmap_list.append(attmap)

        for i in range(self.n_views):
            new_f_list = []
            for j in range(self.n_views):
                new_f = self.gamma[i][j] * \
                    self.mvfrm[i](cur_f_list[i], cur_f_list[j], attmap_list[j])
                new_f_list.append(new_f)
            rec_f = torch.stack(new_f_list, dim=0).sum(dim=0) + cur_f_list[i]
            re_f_list.append(rec_f)
            
        f = rearrange(torch.stack(re_f_list, dim=0),
                      'v b c h w -> (b v) c h w', v=self.n_views)
        atten = rearrange(torch.stack(atten_list, dim=0),
                          'v b (t N) S -> (b v t) N S', v=self.n_views, t=self.video_length)

        ux4 = self.defuse4(f)
        ux3 = self.defuse3(f)
        ux2 = self.defuse2(f)
        ux1 = self.defuse1(f)
        ux0 = self.defuse0(f)

        ux4 = rearrange(x4, '(b v t) c h w -> (b v) t c w h',
                        t=self.video_length, v=self.n_views)[:, 0] + ux4

        u4 = self.up_tr256(ux4, ux3)
        u3 = self.up_tr128(u4, ux2)
        u2 = self.up_tr64(u3, ux1)
        u1 = self.up_tr32(u2, ux0)

        x = self.out_tr(u1)
        # return x
        return x, atten



if __name__ == '__main__':
    model = M2EchoSeg(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.1,
        dropout_dim=2,
        bias=True,
    )
