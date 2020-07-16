# quanlization_pytorch

checkpoint文件夹存放模型的预训练权重  
dataset文件夹存放数据集  
resnet.py 自己写的resnet网络模型  
resnet_original.py torchvision中自带的resnet模型  
resnet_train.py 训练resnet模型  
resnet_quantization.py 将训练好的resnet模型量化，预训练权重由float转为8位二进制表示  
resnet_for_quantization.py 用于量化的自己写的resnet模型
