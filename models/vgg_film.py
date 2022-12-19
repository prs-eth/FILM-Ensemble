'''FiLM-VGG in PyTorch.'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.CBN import CBN2D as CBN2D

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn',]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
}


class VGG(nn.Module):
    def __init__(self, features, task_count, drop_rate, num_classes, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.task_count = task_count

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        B = x.shape[0]

        x = x.repeat_interleave(self.task_count, dim=0)

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        x = x.view(B, self.task_count, -1)

        return x

    def get_layer(self, name):
        return getattr(self, name)

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


def make_layers(cfg, task_count, is_cbn_trainable, cbn_gain, drop_rate, cbn_version, init_type):
    layers = []
    in_channels = 3
    for ix, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            cbn_layer = CBN2D(n_ensemble=task_count, num_features=v, name="cbn"+str(ix),trainable=is_cbn_trainable, cbn_gain=cbn_gain, init_type=init_type)

            dropout_layer = nn.Dropout2d(p=drop_rate)

            layers += [conv2d, cbn_layer, nn.ReLU(inplace=True), dropout_layer]

            in_channels = v
    return nn.Sequential(*layers)


def vgg_cbn(cfg, task_count, pretrained=False, is_cbn_trainable=True, num_classes=10, cbn_gain=1.0, drop_rate=0., cbn_version='v1', init_type='xaiver',  **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(cfg=cfg, task_count=task_count, is_cbn_trainable=is_cbn_trainable,
                            cbn_gain=cbn_gain, drop_rate=drop_rate, cbn_version=cbn_version, init_type=init_type, **kwargs),
                            task_count=task_count,
                            drop_rate=drop_rate,
                            num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model