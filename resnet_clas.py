from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torchvision import datasets, transforms
import transforms
from torch.autograd import Variable
#from torchvision.datasets import ImageFolder
from folder import ImageFolder
from transforms import ToTensor
from PIL import Image
import math

#torch.save()

def png_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
            #return img.convert('RGB')

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def zeroone_transform(x):
    for j in range(224):
        for i in range(224):
            if x[0][i][j] == 1:
                x[0][i][j] = 0
            else:
                x[0][i][j] = 1
    return x

def floor_transform(x):
    return 1.0 - x.floor()

mnist_transformer = transforms.Compose([transforms.ToTensor(),
                              #transforms.Lambda(lambda x: zeroone_transform(x)),
                              transforms.Lambda(lambda x: 1.0 - x.floor()),
                              transforms.Normalize((0.1307,), (0.3081,))
                             ])

train_data = ImageFolder(root='./sample/sample.train.images', loader=png_loader, transform=transforms.Normalize((0.1307,), (0.3081,)))
dev_data = ImageFolder(root='./sample/sample.train.images', loader=png_loader, transform=transforms.Normalize((0.1307,), (0.3081,)))

print(len(train_data))
print(len(dev_data))
'''
for i in range(len(dev_data)):
    x, y = dev_data[i]
    print(x) 
    print(y)
    for m in range(224):
        for n in range(224):
            if math.fabs(x[0][m][n]-2.8214867115) < 1e-5:
                #print("x", end='')
                print("", end='')
            else:
                #print(".", end='')
                print(x[0][m][n], end=' ')
            #print(str(x[0][m][n]) + " ")
        #sys.exit(0)
        print("\n")
    break
'''
train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=args.batch_size, shuffle=True, **kwargs)

dev_loader = torch.utils.data.DataLoader(dev_data,
    batch_size=args.batch_size, shuffle=True, **kwargs)
#dev_loader = train_loader


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #print("basicblock.inplanes: " + str(inplanes))
        #print("planes: " + str(planes))
        #print("stride: " + str(stride))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #print("x.size: " + str(x.size()))
        out = self.conv1(x)
        #print("x.size: " + str(x.size()))
        out = self.bn1(out)
        out = self.relu(out)

        #print("x.size: " + str(x.size()))
        out = self.conv2(out)
        #print("conv2 x.size: " + str(x.size()))
        out = self.bn2(out)

        #print("downsample: " + str(self.downsample))
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SingleChannelResnet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(SingleChannelResnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        #print("planes: " + str(planes))
        #print("blocks: " + str(blocks))
        #print("block.expansion: " + str(block.expansion))
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #print("downsample")
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        #sys.exit(0)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #64, 1, 224, 224
        #print(x.size())
        x = self.conv1(x)
        #64, 64, 112, 112
        #print(x.size())
        x = self.bn1(x)
        #print(x.size())
        x = self.relu(x)
        x = self.maxpool(x)
        #64, 64, 56, 56
        x = self.layer1(x)
        #print("self.layer1: " + str(x.size()))
        x = self.layer2(x)
        #print("self.layer2: " + str(x.size()))
        #sys.exit(0)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
#        x = self.fc(x)
        #print("self.layer2: " + str(x.size()))
        #sys.exit(0)
#        x = F.log_softmax(x)
        return x

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SingleChannelResnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
class STKP(nn.Module):
    def __init__(self, block=BasicBlock, layers=None, num_classes=2, feature_num=4, model=resnet18()):
        self.block = block
        super(STKP, self).__init__()
        self.layers = layers
        self.no_class = num_classes
        self.feat = feature_num
        self.fc = nn.Linear(512 * block.expansion * self.feat, self.no_class)
        self.dmodel = model
        self.wmodel = model
        self.mmodel = model
        self.ymodel = model

    def forward(self,x):
        dout = self.dmodel(x[:, 0:1, :, :])
        wout = self.wmodel(x[:, 1:2, :, :])
        mout = self.mmodel(x[:, 2:3, :, :])
        yout = self.ymodel(x[:, 3:4, :, :])
        out = torch.cat([dout, wout, mout, yout], dim=1)
        out = self.fc(out)
        out = F.log_softmax(out)
        return out

def stockp(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STKP(BasicBlock, [2, 2, 2, 2], 2, 4, resnet18())
    return model

model = stockp(**kwargs)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
opta = optim.Adam(model.parameters(), lr=args.lr, )
def train(epoch):
    model.train()
    data_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data[:,:,222,:])
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        #print(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        data_num += len(data)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, data_num, len(train_loader.dataset),
                100. * data_num / len(train_loader.dataset), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in dev_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(dev_loader.dataset) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dev_loader.dataset),
        100. * correct / len(dev_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
