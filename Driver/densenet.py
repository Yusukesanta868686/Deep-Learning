import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.checkpoint as cp

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function

class _Denselayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient = False):
        super(_Denselayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace = True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size = 1, stride = 1, bias = False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace = True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p = self.drop_rate, training = self.training)
        return new_features
    
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient = False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _Denselayer(
                num_input_features + i * growth_rate,
                growth_rate = growth_rate,
                bn_size = bn_size,
                drop_rate = drop_rate,
                memory_efficient = memory_efficient
            )
            self.add_module('denselayer%d' %(i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)

        return torch.cat(features, 1)
    
class _Trainsition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Trainsition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace = True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size = 1, stride = 1, bias = False))
        self.add_module('pool', nn.AvgPool2d(kernel_size = 2, stride = 2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate = 32, block_config = (6, 12, 24, 16), num_init_features = 256, bn_size = 4, drop_rate = 0, num_classes = 4, memory_efficient = False):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size = 7, stride = 2, padding = 3, bias = False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace = True)),
            ('pool0', nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers = num_layers,
                num_input_features = num_features,
                bn_size = bn_size,
                growth_rate = growth_rate,
                drop_rate = drop_rate,
                memory_efficient = memory_efficient
            )
            self.features.add_module('denseblock%d'%(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i == len(block_config) - 1:
                trans = _Trainsition(num_input_features = num_features,
                                     num_output_features = num_features // 2)
                self.features.add_module('trainsition%d'%(i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def change_cls_number(self, num_classes):
        print("Changing Full Connected")
        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace = True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

