import os
import time
import pandas as pd
import csv
from PIL import Image

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

last_layer_names = ['net.last_linear.weight', 'net.last_linear.bias',
                    'net.classifier.weight', 'net.classifier.bias',
                    'net.fc.weight', 'net.fc.bias',

                    'module.net.last_linear.weight',
                    'module.net.last_linear.bias',
                    'module.net.classifier.weight',
                    'module.net.classifier.bias',
                    'module.net.fc.weight', 'module.net.fc.bias', ]


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
                [normalize(transforms.ToTensor()(crop)) for crop in crops])),
            # .ToTensor() is a class NOT a function!
        ])


class DYDataSet(Dataset):
    def __init__(self, root_dir, img_label_array, transform=None):
        """DYDataSet init

        Args:
            root_dir: data root dir
            img_label_array: 2d array, 1st col contains img name, 2cd col 
            contains label
            transform: torchvision.transform (default: {None})
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


def accuracy(output, label, topk=(1,)):
    """acc compute func

    this func can compute accs from model's output and label

    Args:
        output: model's output in prob
        label: this sample's label
        topk: top'k acc would be computed (default: {(1,)})

    Returns:
        topk's result
        list
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = label.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))

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


def train(model, train_loader, val_loader, train_criterion, val_criterion,
          checkpoint_file, epochs):
    """whole train loop for any model

    this func is the 'trainning' func for any pytorch model. Given model, data_
    loader, criterion(loss), checkpoint_file and train_cpochs, this func will
    train modle with many 'epochs'

    Args:
        model: nn.module, the model to train
        train_loader: train dataloader
        val_loader: validation dataloder
        train_criterion: loss func for train
        val_criterion: loss func for validation
        checkpoint_file: the checkpoint file to save and reload
        epochs: how many epochs the model will train
    """
    min_loss = float("inf")
    lr = 0
    patience = 0

    for epoch in range(epochs):
        print(f'[+] epoch {epoch}')

        if epoch == 1:
            lr = 0.00003
            cnt = 0
            for param in model.parameters():
                param.requires_grad = True
                cnt += 1
            print(f'[+] params to be optimized: {cnt}')

        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load(checkpoint_file))
            lr = lr / 10
            print(f'[*] declining lr={lr}')

        if epoch == 0:
            lr = 0.001
            cnt = 0
            for name, param in model.named_parameters():
                if(name not in last_layer_names):
                    param.requires_grad = False
                else:
                    cnt += 1
            print(f'[+] params to be optimized: {cnt}')

            model = torch.nn.DataParallel(model).cuda()
            optimizer_nfc = None
            optimizer_fc = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        else:
            params_nfc = []
            params_fc = []
            for name, param in model.named_parameters():
                if(name not in last_layer_names):
                    params_nfc.append(param)
                else:
                    params_fc.append(param)

            optimizer_nfc = torch.optim.Adam(
                params=params_nfc, lr=lr, weight_decay=0.0001)
            optimizer_fc = torch.optim.Adam(
                params=params_fc, lr=10*lr, weight_decay=0.0001)

        print(f'[+] lr={lr}')
        train_one_epoch(train_loader, model, train_criterion,
                        epoch, optimizer_fc, optimizer_nfc)

        log_loss = validate(val_loader, model, val_criterion)
        if log_loss < min_loss:
            torch.save(model.state_dict(), checkpoint_file)
            print(f'[+] val loss improved from {min_loss:.5f} to '
                  f'{log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


def train_one_epoch(train_loader, model, criterion, epoch, optimizer_fc,
                    optimizer_nfc=None):
    """just train one epoch in trainning peroid

    This func will be called in func `train` every epoch. We adopt the strategy
    that params have different learning rate between fc layers and non-fc 
    layers, so there have two optimizers 

    Args:
        train_loader: train dataloader
        model: nn.module
        criterion: loss func 
        epoch: which epoch when this func called 
        optimizer_fc: optimizer of fc layer params
        optimizer_nfc: optimizer of non-fc layer params (default: {None})
    """
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
        if(optimizer_nfc is not None):
            optimizer_nfc.zero_grad()
        optimizer_fc.zero_grad()
        loss.backward()
        if(optimizer_nfc is not None):
            optimizer_nfc.step()
        optimizer_fc.step()

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
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, top1=top1, top5=top5))

        print('[+] avg val loss {loss_avg:.3f} Prec@1 {top1.avg:.3f} '
              'Prec@5 {top5.avg:.3f}'.format(loss_avg=losses.avg,
                                             top1=top1, top5=top5))

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
