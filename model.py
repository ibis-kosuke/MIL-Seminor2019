import torch
from torch import nn

class VGGnet(nn.Module):
    def __init__(self, cfg_list, num_classes=100, batch_norm=True, init_weight=True):
        super(VGGnet, self).__init__()
        self.basenet = self.make_basemodule(cfg_list, batch_norm)
        self.classifier = nn.Sequential(
            #4096 -> 2048
            nn.Linear(256*4*4,2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes)
        )
        
        if init_weight:
            self._init_weight()

    def forward(self, input_tensor):
        #input_tensor: batch x 3 x 32 x 32
        batch_size = input_tensor.size(0)
        #feat_map: batch x 256 x 4 x 4
        feat_map = self.basenet(input_tensor)
        #feat_vec: batch x 4096
        feat_vec = feat_map.view(batch_size, -1)

        #activates: real_number of batch x 100 
        activates = self.classifier(feat_vec)

        return activates



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' )##
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            

    def make_basemodule(self,cfg_list=None, batch_norm=True):
        assert isinstance(cfg_list, list) 
        layers = []
        in_channel = 3
        for layer in cfg_list:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channel, layer, kernel_size=3, padding=1)
                in_channel = layer
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)



