import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
def get_inplanes():
    return [64, 128, 256, 512]
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)
        return out
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_fea = x
        x_fea = x_fea.reshape(x.shape[0], -1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, x_fea
def resnet18(num_classes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(),n_classes=num_classes)
def resnet34(num_classes=2):
    return ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(),n_classes=num_classes)
def resnet50(num_classes=2):
    return ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(),n_classes=num_classes)
model = resnet18()
class MeanTeacherNet(nn.Module):
    def __init__(self, lamb = 0.9):
        super().__init__()
        self.lamba = lamb
        self.student1 = model
        self.teacher = model
        for param_s, param_t in zip(self.student1.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False #teachernet paraments will not apply grad
    @torch.no_grad()
    def ema_update(self):
        for param_s, param_t in zip(self.student1.parameters(), self.teacher.parameters()):
            param_t.data = param_t.data * self.lamba + param_s.data * (1. - self.lamba)
            param_t.data = param_s.data
def student2_net(ema = False):
    model_stu2 = resnet18()
    if ema:
        for param in model_stu2.parameters():
            param.detach_()
    return model_stu2
class correctnet(nn.Module):
    def __init__(self, hx_dim, cls_dim, h_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.in_class = self.num_classes
        self.hdim = h_dim
        in_dim = hx_dim + cls_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, num_classes))
    def forward(self, hx, y):
        hin = torch.cat([hx, y], dim = -1)
        logit = self.net(hin)
        out = F.softmax(logit, -1)
        return out
def correct():
    correct = correctnet(16384, 2, 32, 2)
    return correct
def mtnet():
    model = MeanTeacherNet()
    return model
def stu1_loss(pre_stu1_sup, pre_stu1, gl, pre_tea, a):
    # pre_stu1_sup, the samples with gold label (gl)
    loss_s1 = F.cross_entropy(pre_stu1_sup, gl)
    loss_ts1 = F.mse_loss(pre_stu1, pre_tea)
    loss_stu1 = a*loss_ts1 + loss_s1
    return loss_stu1
def stu2_loss(pre_stu1_sup, pre_stu2, new_doc, pre_tea, b, c):
    #new_doc, corrected label
    pre_stu2_sup = pre_stu2[:pre_stu1_sup.size(0)]
    pre_stu2_unsup = pre_stu2[pre_stu1_sup.size(0):]# the samples without gold label
    new_doc_unsup = new_doc[pre_stu1_sup.size(0):]
    loss_s2 = F.cross_entropy(pre_stu2_unsup, new_doc_unsup)
    loss_ts2 = F.mse_loss(pre_stu2, pre_tea)
    loss_s1s2 = F.mse_loss(pre_stu1_sup, pre_stu2_sup)
    loss_stu2 = loss_s2 + b*loss_ts2 + c*loss_s1s2 
    return loss_stu2
def correct_loss(gl, pre_doc):
    # pre_doc, predictions from network, not label
    new_doc_sup = pre_doc[:gl.size(0)]
    loss_rec = F.cross_entropy(gl, new_doc_sup) 
    return loss_rec