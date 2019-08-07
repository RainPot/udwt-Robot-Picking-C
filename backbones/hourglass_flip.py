"""
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) Geo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['HourglassNet', 'Hourglass']


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class ResidualBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.skip_connection = nn.Sequential(
            Downsample(filt_size=1, stride=stride, channels=planes),
            nn.Conv2d(inplanes, planes, (1, 1), stride=1, bias=False),
            nn.BatchNorm2d(planes)
        ) if stride != 1 or inplanes != planes else nn.Sequential()

        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        skip = self.skip_connection(x)
        return self.relu(out + skip)


class ConvBNRelu(nn.Module):
    def __init__(self, kernel_size, inplane, plane, stride=1, with_bn=True, with_relu=True):
        super(ConvBNRelu, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            inplane, plane, (kernel_size, kernel_size),
            padding=(padding, padding), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(plane) if with_bn else nn.Sequential()
        self.with_relu = with_relu
        if with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.with_relu:
            out = self.relu(out)
        return out


class Hourglass(nn.Module):
    def __init__(self, n, inplanes, layer_nums):
        super(Hourglass, self).__init__()
        self.n = n

        cur_layer_num = layer_nums[0]
        nxt_layer_num = layer_nums[1]

        cur_inplane = inplanes[0]
        nxt_inplane = inplanes[1]

        self.up1 = self.make_residual_layer(cur_inplane, cur_inplane, cur_layer_num)
        self.max1 = self.make_pool_layer()
        self.low1 = self.make_hg_layer(cur_inplane, nxt_inplane, cur_layer_num)
        self.low2 = Hourglass(n - 1, inplanes[1:], layer_nums[1:]) if self.n > 1 else \
            self.make_residual_layer(nxt_inplane, nxt_inplane, nxt_layer_num)
        self.low3 = self.make_reverse_residual_layer(nxt_inplane, cur_inplane, cur_layer_num)
        self.up2 = self.make_upsample_layer()

    @staticmethod
    def make_hg_layer(inplane, plane, layer_num):
        layers = [ResidualBlock(inplane, plane, stride=2)]
        layers += [ResidualBlock(plane, plane) for _ in range(layer_num - 1)]
        return nn.Sequential(*layers)

    @staticmethod
    def make_residual_layer(inplane, plane, layer_num, stride=1):
        layers = [ResidualBlock(inplane, plane, stride)]
        for _ in range(1, layer_num):
            layers.append(ResidualBlock(plane, plane, stride))
        return nn.Sequential(*layers)

    @staticmethod
    def make_reverse_residual_layer(inplane, plane, layer_num, stride=1):
        layers = []
        for _ in range(layer_num - 1):
            layers.append(ResidualBlock(inplane, inplane, stride))
        layers.append(ResidualBlock(inplane, plane, stride))
        return nn.Sequential(*layers)

    @staticmethod
    def make_pool_layer():
        return nn.Sequential()
        # TODO: why here don't use pooling.
        # return nn.MaxPool2d(kernel_size=2, stride=2)

    @staticmethod
    def make_upsample_layer():
        # TODO: bilinear or nearest?
        return nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        Upsample = nn.Upsample(size=(up1.size()[2], up1.size()[3]), mode='bilinear', align_corners=True)
        up2 = Upsample(up2)
        return up1 + up2


class HourglassNet(nn.Module):
    # Hourglass model for CenterNet.
    # Pretrained model: `hdfs://192.168.193.1:9000/data/models/geo/hourglass.pth`
    # We can use torch.load() to load the pretrained model.

    def __init__(self, num_stacks=2):
        super(HourglassNet, self).__init__()

        self.inplanes = 128
        self.num_feats = 256
        self.num_stacks = num_stacks

        # I. Build the pre residual layers.
        # TODO: Here, the original centernet don't have maxpooling, they use a residual block with stride 2.
        #       Ver.2 follows the official setting.
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            ResidualBlock(self.inplanes, 2 * self.inplanes, 2),
        )

        # II. Build the hourglass modules.
        n = 5
        inplanes = [256, 256, 384, 384, 384, 512]
        layer_nums = [2, 2, 2, 2, 2, 4]
        self.hgs = nn.ModuleList([
            Hourglass(n, inplanes, layer_nums) for _ in range(num_stacks)
        ])

        self.convs = nn.ModuleList([
            ConvBNRelu(3, inplanes[0], self.num_feats, with_relu=False) for _ in range(num_stacks)
        ])

        self.residual = nn.ModuleList([
            ResidualBlock(inplanes[0], inplanes[0]) for _ in range(num_stacks - 1)
        ])

        self.inter_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inplanes[0], inplanes[0], (1, 1), bias=False),
                nn.BatchNorm2d(inplanes[0])
            ) for _ in range(num_stacks - 1)
        ])
        self.conv_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.num_feats, inplanes[0], (1, 1), bias=False),
                nn.BatchNorm2d(inplanes[0])
            ) for _ in range(num_stacks - 1)
        ])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Hourglass forward.
        :param x: input image.
        :return: list, which includes output feature of each hourglass block.
        """
        pre_feat = self.pre_layer(x)
        outs = []

        for i in range(self.num_stacks):
            feat = self.hgs[i](pre_feat)
            feat = self.convs[i](feat)
            outs.append(feat)
            feat = torch.relu(feat)

            if i < self.num_stacks - 1:
                pre_feat = self.inter_[i](pre_feat) + self.conv_[i](feat)
                pre_feat = self.relu(pre_feat)
                pre_feat = self.residual[i](pre_feat)

        return outs


def flip_load(path):
    pretrained_dict = torch.load(path)
    new_dict = {}

    for k, v in pretrained_dict.items():
        a, b = k, v
        c = a.split('.')
        if c[-3] == 'skip_connection' and c[-2] == '1':
            # print(a[-8])
            c[-2] = '2'
            a = ''
            for strr in c:
                a = a + strr + '.'
            a = a[:-1]

            new_dict[a] = b
            continue

        elif c[-3] == 'skip_connection' and c[-2] == '0':
            # print(a[-8])
            c[-2] = '1'
            a = ''
            for strr in c:
                a = a + strr + '.'
            a = a[:-1]
            new_dict[a] = b
            continue

        new_dict[a] = b

    return new_dict


def hourglass_net(num_stacks=2):
    """
    Make Hourglass Net.
    :param num_stacks: number of stacked blocks.
    :return: model
    """
    model = HourglassNet(num_stacks=num_stacks)

    model_dict = model.state_dict()
    pretrained_dict = flip_load('./hourglass.pth')
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # model.load_state_dict(torch.load('./hourglass.pth'), strict=False)
    return model