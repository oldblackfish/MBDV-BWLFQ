import torch
import torch.nn as nn
import torch.nn.functional as F
from VAN import van_b0, van_b1, van_b2, van_b3
from VSSM import VSSBlock

class ISSB(nn.Module):

    def __init__(self, inchannel, outchannel):
        super(ISSB, self).__init__()

        self.mamba = VSSBlock(inchannel)
        self.conv1_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=2, padding=0)
        self.conv3_1 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(outchannel * 3)

    def forward(self, x):
        x = self.mamba(x)
        x1 = self.conv1_2(x)
        x2 = self.conv3_1(x)
        x3 = self.conv3_2(x)
        x3 = self.conv3_3(x3)
        out = torch.cat((x1, x2, x3), 1)
        out = self.prelu(self.bn(out))
        return out


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.van = van_b0(pretrained=True, num_classes=1)

        self.issb_1 = ISSB(inchannel=32, outchannel=32)
        self.issb_2 = ISSB(inchannel=64, outchannel=64)
        self.issb_3 = ISSB(inchannel=160, outchannel=160)
        self.issb_4 = ISSB(inchannel=256, outchannel=256)

        self.feature_embedding_1 = nn.Linear(32 * 3, 32)
        self.feature_embedding_2 = nn.Linear(64 * 3, 64)
        self.feature_embedding_3 = nn.Linear(160 * 3, 160)
        self.feature_embedding_4 = nn.Linear(256 * 3, 256)

        '''regression'''
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.task_main = nn.Sequential(
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, 1)
        )

        self.task_aux = nn.Sequential(
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, 8)
        )

    def MSFE(self, x):
        x1, x2, x3, x4 = self.van(x)

        x1 = self.issb_1(x1)
        x1 = self.avg_pool(x1)
        x1 = self.flat(x1)
        x1 = self.feature_embedding_1(x1)

        x2 = self.issb_2(x2)
        x2 = self.avg_pool(x2)
        x2 = self.flat(x2)
        x2 = self.feature_embedding_2(x2)

        x3 = self.issb_3(x3)
        x3 = self.avg_pool(x3)
        x3 = self.flat(x3)
        x3 = self.feature_embedding_3(x3)

        x4 = self.issb_4(x4)
        x4 = self.avg_pool(x4)
        x4 = self.flat(x4)
        x4 = self.feature_embedding_4(x4)

        out = torch.cat([x1, x2, x3, x4], 1)
        return out

    def forward(self, x_left_h, x_left_v, x_right_h, x_right_v):
        c_left_h = self.MSFE(x_left_h)
        c_left_v = self.MSFE(x_left_v)
        c_right_h = self.MSFE(x_right_h)
        c_right_v = self.MSFE(x_right_v)

        out = torch.cat([c_left_h, c_left_v, c_right_h, c_right_v], dim=1)
        q = self.task_main(out)
        c = self.task_aux(out)

        return q, c


if __name__ == "__main__":
    net = Network().cuda()
    from thop import profile

    input1 = torch.randn(1, 3, 224, 224).cuda()
    input2 = torch.randn(1, 3, 224, 224).cuda()
    input3 = torch.randn(1, 3, 224, 224).cuda()
    input4 = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(net, inputs=(input1, input2, input3, input4))
    print('   Number of parameters: %.5fM' % (params / 1e6))
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))
