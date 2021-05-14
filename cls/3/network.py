from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torchsummary import summary
import numpy as np
import segmentation_models_pytorch as smp
import math


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=3):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)
        # print(nn.AvgPool2d(kernel_size=28, stride=28)(e4).shape)
        return nn.AvgPool2d(kernel_size=56, stride=56)(e3).flatten(), out
        # return e2.flatten(), out


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])

        self.feature1 = nn.Conv2d(filters[2], 1, kernel_size=1, stride=1, padding=0)
        self.feature2 = nn.Conv2d(1, filters[2], kernel_size=1, stride=1, padding=0)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.feature1(e3)
        d4 = self.feature2(e4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)
        # print(nn.AvgPool2d(kernel_size=28, stride=28)(e4).shape)
        # return nn.AvgPool2d(kernel_size=56, stride=56)(e3).flatten(), out
        return e4.flatten(), out


class UNetMulti(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(UNetMulti, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.feature1 = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.feature2 = nn.Conv2d(1, filters[3], kernel_size=1, stride=1, padding=0)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.feature1(e4)

        a5 = F.adaptive_avg_pool2d(e5, (1, 1)).squeeze(-1).squeeze(-1)
        # print(a5.size(), math.ceil(a5.size()[0] * 0.1))
        # print(torch.mean(a5[:math.ceil(a5.size()[0] * 0.1)], dim=0, keepdim=True).size())
        # print(torch.mean(a5[math.ceil(a5.size()[0] * 0.1):], dim=0, keepdim=True).size())
        # a6 = self.softmax(torch.cat([torch.mean(a5[:math.ceil(a5.size()[0] * 0.1)], dim=0, keepdim=True),
        #                              torch.mean(a5[math.ceil(a5.size()[0] * 0.1):], dim=0, keepdim=True)], 0))
        # a7 = torch.cat([a6, 1 - a6], 1)

        d5 = self.feature2(e5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)
        # print(nn.AvgPool2d(kernel_size=28, stride=28)(e4).shape)
        # return nn.AvgPool2d(kernel_size=56, stride=56)(e3).flatten(), out
        return e5.flatten(), a5, out


class UNetMulti2(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(UNetMulti2, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.feature1 = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 2)
        self.softmax = nn.Softmax(dim=1)
        self.feature2 = nn.Conv2d(1, filters[3], kernel_size=1, stride=1, padding=0)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.feature1(e4)
        a5 = self.linear(e5.squeeze(1).flatten(start_dim=1, end_dim=2))
        # a6 = self.softmax(a5)
        # print(a5.size(), math.ceil(a5.size()[0] * 0.1))
        # print(torch.mean(a5[:math.ceil(a5.size()[0] * 0.1)], dim=0, keepdim=True).size())
        # print(torch.mean(a5[math.ceil(a5.size()[0] * 0.1):], dim=0, keepdim=True).size())
        # a6 = self.softmax(torch.cat([torch.mean(a5[:math.ceil(a5.size()[0] * 0.1)], dim=0, keepdim=True),
        #                              torch.mean(a5[math.ceil(a5.size()[0] * 0.1):], dim=0, keepdim=True)], 0))
        # a7 = torch.cat([a6, 1 - a6], 1)

        d5 = self.feature2(e5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)
        # print(nn.AvgPool2d(kernel_size=28, stride=28)(e4).shape)
        # return nn.AvgPool2d(kernel_size=56, stride=56)(e3).flatten(), out
        return e5.flatten(), self.softmax(a5), out


class UNetMulti3(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(UNetMulti3, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0]).to('cuda:0')
        self.Conv2 = conv_block(filters[0], filters[1]).to('cuda:0')
        self.Conv3 = conv_block(filters[1], filters[2]).to('cuda:1')
        self.Conv4 = conv_block(filters[2], filters[3]).to('cuda:1')

        self.feature1 = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1, padding=0).to('cuda:1')
        self.linear = nn.Linear(14 * 14, 2).to('cuda:1')
        self.softmax = nn.Softmax(dim=1)
        self.feature2 = nn.Conv2d(1, filters[3], kernel_size=1, stride=1, padding=0).to('cuda:1')

        self.Up4 = up_conv(filters[3], filters[2]).to('cuda:2')
        self.Up_conv4 = conv_block(filters[3], filters[2]).to('cuda:2')

        self.Up3 = up_conv(filters[2], filters[1]).to('cuda:2')
        self.Up_conv3 = conv_block(filters[2], filters[1]).to('cuda:2')

        self.Up2 = up_conv(filters[1], filters[0]).to('cuda:3')
        self.Up_conv2 = conv_block(filters[1], filters[0]).to('cuda:1')

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0).to('cuda:1')

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3.to('cuda:1'))

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.feature1(e4)
        # a5 = e5.to('cuda:1')
        a6 = torch.cat([torch.mean(e5.squeeze(1)[:math.ceil(e5.squeeze(1).size()[0] * 0.92)], dim=0, keepdim=True),
                        torch.mean(e5.squeeze(1)[math.ceil(e5.squeeze(1).size()[0] * 0.92):], dim=0, keepdim=True)], 0)
        a7 = self.linear(a6.flatten(start_dim=1, end_dim=2))

        # a5 = F.adaptive_avg_pool2d(e5, (1, 1)).squeeze(-1).squeeze(-1)
        # print(a5.size(), math.ceil(a5.size()[0] * 0.1))
        # print(torch.mean(a5[:math.ceil(a5.size()[0] * 0.1)], dim=0, keepdim=True).size())
        # print(torch.mean(a5[math.ceil(a5.size()[0] * 0.1):], dim=0, keepdim=True).size())
        # a6 = self.softmax(torch.cat([torch.mean(a5[:math.ceil(a5.size()[0] * 0.1)], dim=0, keepdim=True),
        #                              torch.mean(a5[math.ceil(a5.size()[0] * 0.1):], dim=0, keepdim=True)], 0))
        # a7 = torch.cat([a6, 1 - a6], 1)

        d5 = self.feature2(e5)

        d4 = self.Up4(d5.to('cuda:2'))
        d4 = torch.cat((e3.to('cuda:2'), d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2.to('cuda:2'), d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3.to('cuda:3'))
        d2 = torch.cat((e1.to('cuda:3'), d2), dim=1)
        d2 = self.Up_conv2(d2.to('cuda:1'))

        out = self.Conv(d2)

        # d1 = self.active(out)
        # print(nn.AvgPool2d(kernel_size=28, stride=28)(e4).shape)
        # return nn.AvgPool2d(kernel_size=56, stride=56)(e3).flatten(), out
        return e5.flatten(), a7, out


if __name__ == "__main__":
    gpu = True
    devices = [0, 1, 2, 3]
    model = UNetMulti3()
    device = torch.device("cuda:%i" % devices[0]) if gpu else torch.device("cpu")
    # model = torch.nn.DataParallel(model, device_ids=devices)
    # model = model.to(device)

    # model(torch.ones(300, 3, 112, 112).to('cuda:0'))
    summary(model, (3, 112, 112), batch_size=300)
    # print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print("requires_grad: True ", name)
    #     else:
    #         print("requires_grad: False ", name)
