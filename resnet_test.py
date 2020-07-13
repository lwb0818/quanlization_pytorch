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
# from tensorboardX import SummaryWriter


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(description='resnet model for classification--pytorch version')
    # parser.add_argument('--pretrained_model_path', type=str, default='./checkpoint/resnet18_weights.pth')
    parser.add_argument('--pretrained_model_path', type=str, default='./checkpoint/resnet18_quantization_weights.pth')
    args = parser.parse_args()
    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

# start_time = time.time()
# for iteration, (images, targets) in enumerate(trainloader):
#     print('loading time: %f' %(time.time()-start_time))
#     print(images.shape)
#     start_time = time.time()


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


if __name__ == '__main__':
    args = arg_parse()
    model = resnet18(pretrained=True, pretrained_model_path=args.pretrained_model_path, device=device).to(device)
    test_accuracy(model)
