'''FiLM-ResNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .CBN import CBN2D


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, task_count=1, cbn_gain=2.0, is_cbn_trainable=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        # self.bn1 = nn.BatchNorm2d(planes)
        self.cbn1 = CBN2D(n_ensemble=task_count, num_features=planes, name="cbn" + str(1), trainable=is_cbn_trainable,
                          cbn_gain=cbn_gain)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # self.bn2 = nn.BatchNorm2d(planes)
        self.cbn2 = CBN2D(n_ensemble=task_count, num_features=planes, name="cbn" + str(2), trainable=is_cbn_trainable,
                          cbn_gain=cbn_gain)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
                CBN2D(n_ensemble=task_count, num_features=self.expansion * planes, name="cbn_shortcut",
                      trainable=is_cbn_trainable, cbn_gain=cbn_gain)

            )

    def forward(self, x):
        out = F.relu(self.cbn1(self.conv1(x)))
        out = self.cbn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_FILM(nn.Module):
    def __init__(self, block, num_blocks, task_count, cbn_gain, is_cbn_trainable, num_classes=10, init_weights=True):
        super(ResNet_FILM, self).__init__()

        self.task_count = task_count

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.cbn_initial = CBN2D(n_ensemble=task_count, num_features=64, name="cbn_initial", trainable=is_cbn_trainable,
                                 cbn_gain=cbn_gain)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, task_count=task_count, cbn_gain=cbn_gain,
                                       is_cbn_trainable=is_cbn_trainable)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, task_count=task_count, cbn_gain=cbn_gain,
                                       is_cbn_trainable=is_cbn_trainable)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, task_count=task_count, cbn_gain=cbn_gain,
                                       is_cbn_trainable=is_cbn_trainable)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, task_count=task_count, cbn_gain=cbn_gain,
                                       is_cbn_trainable=is_cbn_trainable)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()

    def get_layer(self, name):
        return getattr(self, name)

    def _make_layer(self, block, planes, num_blocks, stride, task_count, cbn_gain, is_cbn_trainable):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, task_count, cbn_gain, is_cbn_trainable))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        B = x.shape[0]
        x = x.repeat_interleave(self.task_count, dim=0)

        out = F.relu(self.cbn_initial(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out_spatial_size = out.shape[-1]
        # out = F.avg_pool2d(out, 4)
        out = F.avg_pool2d(out, out_spatial_size)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        out = out.view(B, self.task_count, -1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def ResNet18_FILM(task_count, cbn_gain, is_cbn_trainable, num_classes):
    return ResNet_FILM(BasicBlock, [2, 2, 2, 2], task_count=task_count, cbn_gain=cbn_gain,
                       is_cbn_trainable=is_cbn_trainable, num_classes=num_classes)


def ResNet34_FILM(task_count, cbn_gain, is_cbn_trainable, num_classes):
    return ResNet_FILM(BasicBlock, [3, 4, 6, 3], task_count=task_count, cbn_gain=cbn_gain,
                       is_cbn_trainable=is_cbn_trainable, num_classes=num_classes)


def ResNet50_FILM(task_count, cbn_gain, is_cbn_trainable, num_classes):
    return ResNet_FILM(Bottleneck, [3, 4, 6, 3], task_count=task_count, cbn_gain=cbn_gain,
                       is_cbn_trainable=is_cbn_trainable, num_classes=num_classes)


def ResNet101_FILM(task_count, cbn_gain, is_cbn_trainable, num_classes):
    return ResNet_FILM(Bottleneck, [3, 4, 23, 3], task_count=task_count, cbn_gain=cbn_gain,
                       is_cbn_trainable=is_cbn_trainable, num_classes=num_classes)


def ResNet152_FILM(task_count, cbn_gain, is_cbn_trainable, num_classes):
    return ResNet_FILM(Bottleneck, [3, 8, 36, 3], task_count=task_count, cbn_gain=cbn_gain,
                       is_cbn_trainable=is_cbn_trainable, num_classes=num_classes)


if __name__ == '__main__':
    num_ensembles = 10
    net = ResNet18_FILM(num_ensembles, 1., True, 10)
    y = net(torch.randn(5, 3, 32, 32))
    print(y.size())