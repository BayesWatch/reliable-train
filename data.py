'''Data loading utility'''
import os

import torch

import torchvision
import torchvision.transforms as transforms

class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return self.parent_ds[i+self.offset]

def cifar10(scratch_loc, minibatch_size):
    minibatch_size = int(minibatch_size)
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_save_loc = os.path.join(scratch_loc, 'data')
    print("Data saved to: %s"%data_save_loc)
    trainvalset = torchvision.datasets.CIFAR10(root=data_save_loc, train=True, download=True, transform=transform_train)
    trainset = PartialDataset(trainvalset, 0, 40000)
    #trainset = PartialDataset(trainvalset, 0, 2*minibatch_size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True, num_workers=2)

    valset = PartialDataset(trainvalset, 40000, 10000)
    #valset = PartialDataset(trainvalset, 40000, 200)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_save_loc, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, valloader, testloader
