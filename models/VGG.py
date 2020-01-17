'''VGG11/13/16/19 in Pytorch.'''
import torch
from torch import nn, sigmoid
import math
import numpy as np

from utils.tools import my_print

model_archs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, vgg_name, in_channels, final_classes, init_weights=True, batch_norm=True):
        super(VGG, self).__init__()

        self.features = make_layers(model_archs[vgg_name], in_channels, batch_norm)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, final_classes),
        )

        self.final_activation = sigmoid

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float()
        x = self.features(x)
        #print("Size of features:", x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        y = self.final_activation(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(model_archs, in_channels, batch_norm=False):
    layers = []

    for v in model_archs:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



if __name__=='__main__':
    model=VGG(vgg_name='VGG16', in_channels=3, final_classes=2, init_weights=True, batch_norm=True)
    x=np.random.randint(0, 255, (1,3, 224, 224))
    x=torch.tensor(x).float()
    pred=model.forward(x)
    my_print("Numer of parameters: {}".format(sum(p.numel() for p in model.parameters())))

