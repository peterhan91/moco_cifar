import os
import json
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from args import parser
from loader import CIFAR10Pair
from moco.moco_net import ModelMoCo
from train import train
from test import test

def main(args):
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    # data prepare
    train_data = CIFAR10Pair(root='data', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                                num_workers=16, pin_memory=True, drop_last=True)

    memory_data = CIFAR10(root='data', train=True, transform=test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, 
                                num_workers=16, pin_memory=True)

    test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, 
                                num_workers=16, pin_memory=True)

    # create model
    model = ModelMoCo(
            dim=args.moco_dim,
            K=args.moco_k,
            m=args.moco_m,
            T=args.moco_t,
            arch=args.arch,
            bn_splits=args.bn_splits,
            symmetric=args.symmetric,
        ).cuda()
    print(model.encoder_q)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), 
                            lr=args.lr, weight_decay=args.wd, momentum=0.9)

    # load model if resume
    epoch_start = 1
    if args.resume is not '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))

    # logging
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    # training loop
    for epoch in range(epoch_start, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, args)
        results['train_loss'].append(train_loss)
        test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
        results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
        # save model
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, 
                    args.results_dir + '/model_last.pth')


if __name__ == '__main__':
    args = parser()
    main(args)