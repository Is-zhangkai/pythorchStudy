import torchvision
from PIL import Image
from torch.utils.data import DataLoader

#使用datasets数据集
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# train_set=torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)

# 使用dataloader加载数据
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

img, tar = test_set[0]
print(img.size)
print(tar)

for data in test_loader:
    img, tar = data
    print(img.size)
    print(tar)


