import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from resnet import resnet18
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(description='resnet model for classification--pytorch version')
    parser.add_argument('--pretrained_model_path', type=str, default='')
    args = parser.parse_args()
    return args


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
batchsize = 64
LR = 0.1
num_epoches = 200
step_size=30
gamma=0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # padding后随机裁剪
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

# start_time = time.time()
# for iteration, (images, targets) in enumerate(trainloader):
#     print('loading time: %f' %(time.time()-start_time))
#     print(images.shape)
#     start_time = time.time()

writer = SummaryWriter(log_dir='scalar')


def train_accuracy(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            img, labels = data
            img, labels = img.to(device), labels.to(device)
            out = model(img)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('Accuracy of the network on the train image: %d %%' % (100 * correct / total))
    return 100.0 * correct / total


def test_accuracy(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            img, labels = data
            img, labels = img.to(device), labels.to(device)
            out = model(img)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('Accuracy of the network on the 10000 test image: %d %%' % (100 * correct / total))
    return 100.0 * correct / total


def train():
    model = resnet18().to(device)
    # for name, para in model.named_parameters():
    #     print(name)
    #     print(para.shape)

    if torch.cuda.is_available():
        cudnn.benchmark = True
        print('cudnn works')

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    #    optimizer = optim.SGD(model.parameters(), lr = LR, momentum=0.9)
    #    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    #    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    iteration = 0
    for epoch in range(num_epoches):
        scheduler.step()  # 学习率更新
        for i, data in enumerate(trainloader):
            time_start = time.time()
            iteration = iteration + 1
            img, labels = data
            img, labels = img.to(device), labels.to(device)

            output = model(img)
            loss = criterion(output, labels).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('scalar/loss', loss.item(), iteration)
            print('Epoch: %d, Iteration: %d, lr:%f, loss:%f, time: %.3f' %
                  (epoch, iteration, optimizer.param_groups[0]['lr'], loss.data.cpu(), (time.time() - time_start)))

        writer.add_scalar('scalar/train_accuracy', train_accuracy(model), epoch + 1)
        writer.add_scalar('scalar/test_accuracy', test_accuracy(model), epoch + 1)

    torch.save(model.state_dict(), './checkpoint/resnet18_weights.pth')


if __name__ == '__main__':
    args = arg_parse()
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')
    print('Training starts')
    train()
    writer.close()
    print('Training Finished')
