import torch
import torch.nn as nn
import torch.nn.functional as F
class cnn_Network(nn.Module):
    def __init__(self, out_dim):
        super(cnn_Network, self).__init__()
        self.conv1 = conv1()
        self.senet1 = sent(16,reduction=4,size = (77,66))
        self.senet2 = sent(32, reduction=4, size = (39,33))
        self.senet3 = sent_raw(64, reduction=4)
        self.down1 = down(in_channels=16, out_channels=32, strides=2)
        self.down2 = down(in_channels=32, out_channels=64, strides=2)
        self.BAM1 = BAM(channels=32, reduction=4)
        self.BAM2 = BAM(channels=64, reduction=4)
        self.con4 = conv4_x()
        self.con5 = conv5_x()
        self.avg = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(1280,64)
        self.fc2 = nn.Linear(64, out_dim)
        self.drop = nn.Dropout(0.7)


    def forward(self, x):
        con1 = self.conv1(x)

        con2 = self.senet1(con1)
        down1 = self.down1(con2)
        con4 = self.BAM1(down1)

        con4 = self.con4(con4)

        con4 = self.senet2(con4)
        down2 = self.down2(con4)
        con5 = self.BAM2(down2)

        con5 = self.con5(con5)
        con5 = self.senet3(con5)

        avg = self.avg(con5)
        flatten = nn.Flatten(1)(avg)
        dense1 = self.fc1(flatten)
        dense1 = self.drop(dense1)
        dense = self.fc2(dense1)
        out = dense
        return out
class conv1(nn.Module):
    def __init__(self):
        super(conv1, self).__init__()
        self.zpad1 = nn.ZeroPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.ac = nn.ReLU()
        self.zpad2 = nn.ZeroPad2d(1)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.zpad1(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.ac(x)
        x = self.zpad2(x)
        x = self.mp(x)
        return x
class sent(nn.Module):
    def __init__(self,channels=16, reduction=4, size=(77, 66)):
        super(sent, self).__init__()

        self.conv1 = nn.Conv2d(in_channels= channels, out_channels= channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=0)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.hardsigmoid = nn.Hardsigmoid()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1)
        )
        self.size = size

    def forward(self, x):

        avg_x = self.avg_pool(x)
        avg_x = self.fc(avg_x)
        max_x = self.max_pool(x)
        max_x = self.fc(max_x)
        cbam_feature = avg_x+max_x
        cbam_feature = self.hardsigmoid(cbam_feature)
        cbam_feature = torch.mul(x, cbam_feature)

        x1 = self.conv1(x)
        avg_x1 = self.avg_pool(x1)
        avg_x1 = self.fc(avg_x1)
        max_x1 = self.max_pool(x1)
        max_x1 = self.fc(max_x1)
        cbam_feature1 = avg_x1+max_x1
        cbam_feature1 = self.hardsigmoid(cbam_feature1)
        cbam_feature1 = torch.mul(x1, cbam_feature1)
        cbam_feature1 = F.interpolate(cbam_feature1, size= self.size, mode='bilinear', align_corners=True)


        x2 = self.conv2(x)
        avg_x2 = self.avg_pool(x2)
        avg_x2 = self.fc(avg_x2)
        max_x2 = self.max_pool(x2)
        max_x2 = self.fc(max_x2)
        cbam_feature2 = avg_x2+max_x2
        cbam_feature2 = self.hardsigmoid(cbam_feature2)
        cbam_feature2 = torch.mul(x2, cbam_feature2)
        cbam_feature2 = F.interpolate(cbam_feature2, size= self.size, mode='bilinear', align_corners=True)

        cbam_all = cbam_feature + cbam_feature1 + cbam_feature2

        return cbam_all

class sent_raw(nn.Module):
    def __init__(self,channels=16, reduction=4):
        super(sent_raw, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.hardsigmoid = nn.Hardsigmoid()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1)
        )


    def forward(self, x):
        avg_x = self.avg_pool(x)
        avg_x = self.fc(avg_x)

        max_x = self.max_pool(x)
        max_x = self.fc(max_x)

        cbam_feature = avg_x+max_x

        cbam_feature = self.hardsigmoid(cbam_feature)

        return torch.mul(x,cbam_feature)
class BAM(nn.Module):
    def __init__(self, channels=32, reduction=4):
        super(BAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels,int(channels/reduction),kernel_size=1, stride=1),
            nn.Conv2d(int(channels/reduction),channels,kernel_size=1, stride=1),
            nn.BatchNorm2d(channels)
        )
        self.cbam = nn.Sequential(
            nn.Conv2d(channels, int(channels / reduction),kernel_size=1, stride=1 ),
            nn.Conv2d(int(channels / reduction),int(channels / reduction), kernel_size=3, stride=1,padding="same"),
            nn.Conv2d(int(channels / reduction), int(channels / reduction), kernel_size=3, stride=1,padding="same"),
            nn.Conv2d(int(channels / reduction), 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B,C,H,W = x.shape
        channel = self.avg_pool(x)
        channel = self.fc(channel)
        channel = channel.repeat(1,1,H,W)
        spat = self.cbam(x)
        spat = spat.repeat_interleave(C, 1)
        sum = channel+spat
        bam = self.sigmoid(sum)
        mul = torch.mul(x,bam)
        return mul
class conv4_x(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, strides=1):
        super(conv4_x, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if strides == 2:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
            self.bn_downsample = nn.BatchNorm2d(out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn_downsample(identity)
        out += identity
        out = self.relu(out)
        return out
class conv5_x(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, strides=1):
        super(conv5_x, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if strides == 2:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
            self.bn_downsample = nn.BatchNorm2d(out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out

class down(nn.Module):
    def __init__(self, in_channels=16, out_channels=32, strides=2):
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)

    def forward(self, x):
        con = self.conv1(x)

        return con