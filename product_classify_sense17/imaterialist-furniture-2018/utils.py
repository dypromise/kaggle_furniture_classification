import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import csv
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.autograd import Variable


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

last_layer_names = ['net.last_linear.weight', 'net.last_linear.bias',
                    'net.classifier.weight', 'net.classifier.bias']

l2_reg_param_names = ['module.net.last_linear.weight', 'module.net.last_linear.bias',
                      'module.net.classifier.weight', 'module.net.classifier.bias']


def get_transforms(mode='train', input_size=224, resize_size=256):
    if(mode == 'train'):
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # （-10， 10）degree rotation range
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor(),
            normalize
        ])
    elif(mode == 'valid'):
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif(mode == 'test'):
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.TenCrop(input_size),
            transforms.Lambda(lambda crops: torch.stack(
                [normalize(transforms.ToTensor()(crop)) for crop in crops])),  # .ToTensor() is a class NOT a function!
        ])


class DYDataSet(Dataset):

    def __init__(self, root_dir, img_label_array, transform=None):
        """Our own dataset class.
        Arguments:
            root_dir {str} -- image data root dir.
            img_label_array {array} -- 2D array
        """
        self.img_frames = img_label_array
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_frames)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.img_frames[idx, 0])
        img = Image.open(img_name).convert('RGB')
        label = int(self.img_frames[idx, 1])

        if self.transform:
            img = self.transform(img)
        return (img, label)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


def get_l2_regularization(variable_list):
    l2_reg = 0
    for W in variable_list:
        if l2_reg == 0:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def train(model, train_loader, val_loader, criterion, checkpoint_file, epochs=30):
     # training loop
    min_loss = float("inf")
    lr = 0
    patience = 0
    for epoch in range(epochs):
        print(f'[+] epoch {epoch}')
        if epoch == 1:
            lr = 0.0001
            print(f'[+] set lr={lr}')
            cnt = 0
            for param in model.parameters():
                param.requires_grad = True
                cnt += 1
            print(f'params to be optimized: {cnt}')

        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load(checkpoint_file))
            lr = lr / 3
            print(f'[+] set lr={lr}')

        if epoch == 0:
            lr = 0.001
            print(f'[+] set lr={lr}')
            cnt = 0
            for name, param in model.named_parameters():
                if(not name in last_layer_names):  # resnet
                    param.requires_grad = False
                else:
                    cnt += 1
            print(f'[+] params to be optimized: {cnt}')
            # Training with multi GPUs
            model = torch.nn.DataParallel(model).cuda()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0002)

        # train for one epoch
        train_one_epoch(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        log_loss = validate(val_loader, model, criterion)

        if log_loss < min_loss:
            torch.save(model.state_dict(), checkpoint_file)
            print(f'[+] val loss improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train().cuda().float()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda().float()
        target = target.view(-1).cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda().float()
            target = target.view(-1).cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

        print('[+] avg val loss {loss_avg:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(loss_avg=losses.avg, top1=top1, top5=top5))

    return losses.avg


def complement_prediction(test_whole_file, preds_csv, new_csv):
    with open(new_csv, 'w') as f3:

        f1 = open(test_whole_file, 'r')
        preds_frame = pd.read_csv(preds_csv)

        new_csv_writer = csv.writer(f3, delimiter=',')
        new_csv_writer.writerow(['id', 'predicted'])

        for data in preds_frame.values:
            new_csv_writer.writerow(data)

        test_ids = f1.readlines()
        print(len(test_ids))
        for idx in test_ids:
            idx = int(idx.strip())
            if idx not in preds_frame.values[:, 0]:
                new_csv_writer.writerow([idx, 1])
        f1.close()
