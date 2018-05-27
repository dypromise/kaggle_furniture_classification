from functools import partial
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as M

import utils
import numpy as np
import pandas as pd
import csv
import re
from sklearn.cross_validation import KFold

import sys
import os
import pretrainedmodels
from models.inceptionv4 import inceptionv4


NB_CLASSES = 128


class FinetuneModel(nn.Module):

    def __init__(self, model_name, num_classes, net_cls, net_kwards):
        super().__init__()

        original_model = net_cls(**net_kwards)

        if(model_name.startswith('res')):
            self.features = nn.Sequential(
                *list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(original_model.fc.in_features, num_classes))

        elif(model_name.startswith('dense')):
            self.features = nn.Sequential(
                *list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(original_model.classifier.in_features, num_classes))
        elif(model_name.startswith('dpn')):
            self.net = original_model
            self.net.classifier = nn.Conv2d(
                self.net.classifier.in_channels, num_classes, kernel_size=1, bias=True)
        elif(model_name.startswith('senet') or model_name.startswith('nasnet') or):
            self.net = original_model
            self.net.last_linear = nn.Linear(
                self.net.last_linear.in_features, num_classes)
        else:
            raise Exception('no match model name!')

    def forward(self, x):
        if(model_name.startswith('resnet')):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        elif(model_name.startswith('dense')):
            x = self.features(x)
            x = torch.nn.functional.relu(x, inplace=True)
            x = torch.nn.functional.avg_pool2d(
                x, kernel_size=7).view(x.size(0), -1)
            x = self.classifier(x)
        else:
            x = self.net(x)

        return x


model_dict = {

    'resnet152': partial(FinetuneModel, 'resnet152', NB_CLASSES, M.resnet152),
    'inceptionresnetv2': partial(FinetuneModel, 'inceptionresnetv2', NB_CLASSES, pretrainedmodels.inceptionresnetv2),
    'dpn98': partial(FinetuneModel, 'dpn98', NB_CLASSES, pretrainedmodels.dpn98),
    'dpn131': partial(FinetuneModel, 'dpn131', NB_CLASSES, pretrainedmodels.dpn131),
    'nasnet': partial(FinetuneModel, 'nasnet', NB_CLASSES, pretrainedmodels.nasnetalarge),
    'senet154': partial(FinetuneModel, 'senet154', NB_CLASSES, pretrainedmodels.senet154),
    'densenet201': partial(FinetuneModel, 'densenet201', NB_CLASSES, M.densenet201),
    'densenet161': partial(FinetuneModel, 'densenet161', NB_CLASSES, M.densenet161),
    'densenet169': partial(FinetuneModel, 'densenet169', NB_CLASSES, M.densenet169),
    'inceptionv4': partial(FinetuneModel, 'inceptionv4', NB_CLASSES, pretrainedmodels.inceptionv4),
    'xception': partial(FinetuneModel,  'xception', NB_CLASSES, pretrainedmodels.xception)
}

net_kwards = [{'pretrained': 'imagenet'}, {'pretrained': None}]


def get_model(model_name: str, pretrained=True):
    print('[+] getting model architecture... ')
    model = None
    if(pretrained):
        model = model_dict[model_name](net_kwards[0])
    else:
        model = model_dict[model_name](net_kwards[1])
    print('[+] done.')
    return model


def load_model(model, checkpoint_pth):
    print('[+] loading model parameters...')
    state_dict = torch.load(checkpoint_pth)
    model.load_state_dict(state_dict)
    print('[+] done.')


def load_model_multiGPU(model, checkpoint_pth):
    print('[+] loading model(Multi-GPUs) parameters...')
    a = torch.load(checkpoint_pth)
    import collections
    b = collections.OrderedDict()
    for name, param in a.items():
        b[name[7:]] = param

    model.load_state_dict(b)
    print('[+] done.')


class DY_Model(object):

    def __init__(self, model_name, num_classes=NB_CLASSES, checkpoint_file='', batch_size=64, input_size=224, add_size=32):

        self.num_classes = num_classes
        self.model = None
        self.model_name = model_name
        self.checkpoint_file = checkpoint_file

        self.input_size = input_size
        self.batch_size = batch_size
        self.add_size = add_size

        print('[+] Model Infomation --->\n'
              '\t\tModel name: {0:s}\n'
              '\t\tBatch size: {1:d}\n'
              '\t\tCheckpoint file: {2:s}\n'
              '\t\tInput size: {3:d}\n'
              '\t\tData augmentation: {4:s}'.format(self.model_name, self.batch_size, self.checkpoint_file, self.input_size, str(True)))

    def train_single_model(self, train_dir, train_csv, val_dir, val_csv, epochs):

        train_part = pd.read_csv(train_csv).values  # array type
        val_part = pd.read_csv(val_csv).values

        train_dataset = utils.DYDataSet(
            train_dir,
            train_part,
            utils.get_transforms(
                mode='train', input_size=self.input_size, resize_size=self.input_size+42)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            sampler=None)
        val_dataset = utils.DYDataSet(
            val_dir,
            val_part,
            utils.get_transforms(mode='valid', input_size=self.input_size, resize_size=self.input_size+42))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

        print('[+] trainning with total %d images' % len(train_dataset))

        self.model = get_model(self.model_name, pretrained=True)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        utils.train(self.model, train_loader, val_loader, criterion,
                    checkpoint_file=self.checkpoint_file, epochs=epochs)

    def test_single_model(self, checkpoint_file, test_dir, test_csv, prediction_file_path='test_prediction.npy', ten_crop=False, prob=False):

        print('[+] checkpoint file:{0:s}'.format(checkpoint_file))
        transform = utils.get_transforms(
            mode='valid', input_size=self.input_size, resize_size=self.input_size+self.add_size)

        if(ten_crop):
            print('[+] Using Ten-Crop Testting strategy')
            transform = utils.get_transforms(
                mode='test', input_size=self.input_size, resize_size=self.input_size+self.add_size)

        # get the data part of pd.DataFrame object
        test_array = pd.read_csv(test_csv).values
        test_dataset = utils.DYDataSet(
            test_dir,
            test_array,
            transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

        self.model = get_model(self.model_name, pretrained=False)
        load_model_multiGPU(self.model, checkpoint_file)
        # load_model(self.model, checkpoint_file)

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.eval()

        all_idxs = []
        all_labels = []
        with torch.no_grad():
            print('testting total %d images' % len(test_dataset))
            for i, (input, labels) in enumerate(test_loader):  # tensor type
                print('testting batch: %d/%d' %
                      (i, len(test_dataset)/self.batch_size))

                input = input.cuda()

                if(ten_crop):
                    bs, ncrops, c, h, w = input.size()
                    input = input.view(-1, c, h, w)
                    output = self.model(input).view(
                        bs, ncrops, -1).mean(1).view(bs, -1)  # view to 2-D tensor
                else:
                    output = self.model(input)  # 2-D tensor

                if(not prob):
                    pred = output.topk(1)[-1]  # pytorch tensor type
                else:
                    pred = output

                all_idxs.append(labels)
                all_labels.append(pred.data.cpu())

        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_idxs = torch.cat(all_idxs, dim=0).numpy().reshape(-1, 1)
        res = np.concatenate((all_idxs, all_labels), axis=1)
        print('writing pred file %s ...' % prediction_file_path)
        np.save(prediction_file_path, res)
        print('done.')
