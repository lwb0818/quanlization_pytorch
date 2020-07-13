import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from resnet_for_quantization import resnet18
from torch.quantization import QuantStub, DeQuantStub, QConfig
import torch.quantization
import torch.utils.data
import torchvision.datasets

# Official utils
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, eval_batches=10000, cpu=False):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image.cuda() if not cpu else image)
            cnt += 1
            acc1, acc5 = accuracy(output, target.cuda() if not cpu else target, topk=(1, 5))
            print('.', end='')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt > eval_batches:
                return top1, top5
    return top1, top5


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


# Append input/output quant/dequant stub.
def replace_forward(module):
    module.quant = QuantStub()
    module.dequant = DeQuantStub()
    raw_forward = module.forward

    def forward(x):
        x = module.quant(x)
        x = raw_forward(x)
        x = module.dequant(x)
        return x
    module.forward = forward


cpu = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform_calibration = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # padding后随机裁剪
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
calibration_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform_calibration)
calibrationloader = torch.utils.data.DataLoader(calibration_dataset, batch_size=256, shuffle=True)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

print('the Size (MB) of original resnet18:', os.path.getsize("./checkpoint/resnet18_weights.pth")/1e6)
resnet18_model = resnet18(pretrained=True, pretrained_model_path='./checkpoint/resnet18_weights.pth')
# necessary
resnet18_model.eval()
print('resnet18.layer1: Before fusion \n', resnet18_model.layer1)

replace_forward(resnet18_model)
# top1, top5 = evaluate(resnet18_model, testloader, cpu=cpu)
# print('Evaluation accuracy before quantization and fusion: %2.2f' % top1.avg)

# Do NOT fuse inplaced relu
torch.quantization.fuse_modules(
    resnet18_model,
    [['conv1', 'bn1'],  # 'relu'],
     ['layer1.0.conv1', 'layer1.0.bn1'],  # 'layer1.0.relu'],
     ['layer1.0.conv2', 'layer1.0.bn2'],
     ['layer1.1.conv1', 'layer1.1.bn1'],  # 'layer1.1.relu'],
     ['layer1.1.conv2', 'layer1.1.bn2'],

     ['layer2.0.conv1', 'layer2.0.bn1'],  # 'layer2.0.relu'],
     ['layer2.0.conv2', 'layer2.0.bn2'],
     ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
     ['layer2.1.conv1', 'layer2.1.bn1'],  # 'layer2.1.relu'],
     ['layer2.1.conv2', 'layer2.1.bn2'],

     ['layer3.0.conv1', 'layer3.0.bn1'],  # 'layer3.0.relu'],
     ['layer3.0.conv2', 'layer3.0.bn2'],
     ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
     ['layer3.1.conv1', 'layer3.1.bn1'],  # 'layer3.1.relu'],
     ['layer3.1.conv2', 'layer3.1.bn2'],

     ['layer4.0.conv1', 'layer4.0.bn1'],  # 'layer4.0.relu'],
     ['layer4.0.conv2', 'layer4.0.bn2'],
     ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
     ['layer4.1.conv1', 'layer4.1.bn1'],  # 'layer4.1.relu'],
     ['layer4.1.conv2', 'layer4.1.bn2'],
     ], inplace=True
)
print('resnet18.layer1: After fusion \n', resnet18_model.layer1)


top1, top5 = evaluate(resnet18_model, testloader, cpu=cpu)
print('Evaluation accuracy before quantization: %2.2f' % top1.avg)

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
# q_backend = "qnnpack"
# qconfig = torch.quantization.get_default_qconfig(q_backend)
# torch.backends.quantized.engine = q_backend
# resnet18_model_fusion.qconfig = qconfig
resnet18_model.qconfig = torch.quantization.default_qconfig
print("resnet18_model_fusion.qconfig: \n", resnet18_model.qconfig)
torch.quantization.prepare(resnet18_model, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n resnet18.layer1: After observer insertion \n', resnet18_model.layer1)

# Calibrate with the training set
evaluate(resnet18_model, calibrationloader, eval_batches=10, cpu=cpu)
print('\n Post Training Quantization: Calibration done')

# Convert to quantized model
resnet18_model = resnet18_model.cpu()
torch.quantization.convert(resnet18_model, inplace=True)
print('Post Training Quantization: Convert done')
print('\n resnet18.layer1: After fusion and quantization, note fused modules: \n', resnet18_model)


print("Size of model after quantization")
print_size_of_model(resnet18_model)


# def compute_accuracy(model):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             img, labels = data
#             img, labels = img.to(device), labels.to(device)
#             out = model(img)
#             _, pred = torch.max(out.data, 1)
#             total += labels.size(0)
#             correct += (pred == labels).sum().item()
#     print('Accuracy of the network on the 10000 test image: %d %%' % (100 * correct / total))
#     return 100.0 * correct / total
#
# compute_accuracy(resnet18_model_quantization)

torch.save(resnet18_model.state_dict(), './checkpoint/resnet18_quantization_weights.pth')
resnet18_model.load_state_dict(
    torch.load('./checkpoint/resnet18_quantization_weights.pth', map_location="cpu"))
resnet_state_dict = resnet18_model.state_dict()
top1, top5 = evaluate(resnet18_model, testloader, cpu=True)
print('Evaluation accuracy %2.2f' % top1.avg)
