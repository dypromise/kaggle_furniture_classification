
import os
import numpy as np
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.cross_validation import train_test_split
import torch.nn.functional as f

import utils

data_root = '/mnt/lustre17/yangkunlin/fur_dy/data/'
test_whole_file = os.path.join(data_root, 'whole_test.txt')

train_list = ['res152_val_pred.npy',
              'incepresv2_val_pred.npy',
              'inceptionv4_dp0.2_val_pred.npy',
              'inceptionv4_l2reg0.01_val_pred.npy',
              'inceptionv4_val_pred.npy',
              'xception_val_pred.npy',
              'xception_l2reg0.01_val_pred.npy',
              'xception_dp0.2_l2reg0.01_val_pred.npy',
              'dpn98_val_pred.npy',
              'dpn131_val_pred.npy',
              'nasnet_val_pred.npy',
              'senet154_val_pred.npy'
              ]

test_list = [

    'dense161_ck6.npy..npy',     # 'inceptionResnetv2_ck2.npy..npy',
    'resnext101_32x4d_ck1.npy..npy',
    # 'inceptionv4_ck1.npy..npy',     # 'resnext101_64x4d_ck1.npy..npy',
    'dense169_ck7.npy..npy',
    'dpn107_ck1.npy..npy',     # 'inceptionv4_ck2.npy..npy',
    'senet154_ck2.npy..npy',
    'dpn131_ck1.npy..npy',
    'inceptionv4_dp0.2_l2reg0.01_pred.npy',
    'se_resnet152_ck1.npy..npy',
    'dpn92_ck1.npy..npy',
    'inceptionv4_dp0.2_pred.npy',
    'se_resnext101_32x4d_ck1.npy..npy',     # 'dpn92_ck2.npy..npy',
    'inceptionv4_l2reg0.01_pred.npy',     # 'xception_ck1.npy..npy',
    'dpn98_ck1.npy..npy',
    'nasnet_ck1.npy..npy',
    'xception_dp0.2_l2reg0.01_pred.npy',     # 'dpn98_ck2.npy..npy',
    'res152_ck5.npy..npy',
    'xception_l2reg0.01_pred.npy',
    # 'res152_ck7.npy..npy',     'xception_pred.npy'
    'inceptionResnetv2_ck1.npy..npy',
]

p_list = [os.path.join('/mnt/lustre17/yangkunlin/fur_dy/fur_res', pred_name)
          for pred_name in test_list]


best_checkpoint_file = '/mnt/lustre17/yangkunlin/fur_dy/data/weighted_ensamble_best.pth'
final_preds_csv = '/mnt/lustre17/yangkunlin/fur_dy/data/weighted_ensamble.csv'
num_classes = 128
batch_size = 64


class MyDataset(Dataset):
    def __init__(self, X, labels):
        self.length = len(X)
        self.X = X
        self.labels = labels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.X[idx], self.labels[idx])


class WeightedEnsambleModel(nn.Module):
    """ simple model to weighted ensamble the predictions from first stage
    Extends:
        nn.Module
    """

    def __init__(self, num_classes, num_models):
        super(WeightedEnsambleModel, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes

        self.normalize = torch.nn.Softmax()
        self.weights = torch.nn.Parameter(torch.rand(
            (num_classes, num_models), requires_grad=True))
        self.ones = torch.ones((self.num_models, 1))

    def forward(self, x):
        x = torch.matmul(
            x*self.normalize(self.weights, dim=1), self.ones).squeeze(-1)
        return x


def load_data(preds_list, mode='train'):
    """ load data from npys, each npy file contains predictions of one model
    Keyword Arguments:
        preds_list，mode {str} -- [description] (default: {'train'})
    Returns:
        [turple] -- if mode is `train`, return train_test_split data parts; if mode is `test`, return all data
    """

    X = []
    for i, pred in enumerate(preds_list):
        arr = np.load(pred)
        if(i == 0):
            labels = np.array(arr[:, 0], dtype='int64').reshape((-1, 1))
        X.append(np.array(arr[:, 1:], dtype='float64'))

    # M * N * num_classes -> N * C * M
    X = np.transpose(np.array(X, dtype='float64'), (1, 2, 0))
    if(mode == 'train'):
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2)
        return X_train, y_train, X_test, y_test
    elif(mode == 'test'):
        return X, labels
    else:
        raise Exception(
            "Attribute error: `mode` can only be either `train` or `test`")


def train_ensamble():
    """ train func of weighted ensamble
    """

    X_train, y_train, X_test, y_test = load_train_data(
        preds_list=p_list, mode='train')
    train_dataset = MyDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None)
    val_dataset = MyDataset(X_test, y_test)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    print(f'[+] trainning with {len(train_dataset)} samples, '
          f' validation with {len(val_dataset)} samples')

    model = WeightedEnsambleModel(num_classes, len(p_list))
    criterion = torch.nn.CrossEntropyLoss().cuda()

    EPOCHS = 100
    min_loss = float("inf")
    lr = 0.001
    patience = 0

    for epoch in range(EPOCHS):
        print(f'[+] epoch {epoch}')

        if patience == 3:
            patience = 0
            model.load_state_dict(torch.load(best_checkpoint_file))
            lr /= 3
            print(f'[+] set lr={lr}')

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # train for one epoch
        utils.train_one_epoch(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set after one epoch
        log_loss = utils.validate(val_loader, model, criterion)

        if log_loss < min_loss:
            torch.save(model.state_dict(), best_checkpoint_file)
            print(f'[+] lr = {lr}, val loss improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1

    print(f'[*] trainning done with {EPOCHS} epochs')


def weighted_ensamble(preds_list, test_whole_file, new_csv):
    """ this func is to do weighted ensamble when func `train_ensamble` has been run.
    Arguments:
        preds_list {[npy]} -- the first satge preditions npy file list
        test_whole_file {[str]} -- complete test data list file in csv format
        new_csv {[str]} --  final submittion csv file
    """

    X, idxs = load_test_data(preds_list, mode='test')
    test_dataset = MyDataset(X, idxs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=None)

    model = WeightedEnsambleModel(num_classes, len(p_list))
    model.load_state_dict(torch.load(best_checkpoint_file))
    model.cuda().eval()

    all_idxs = []
    all_preds = []
    with torch.no_grad():
        print('testting total %d images' % len(test_dataset))
        for i, (input, labels) in enumerate(test_loader):  # tensor type

            print('testting batch: %d/%d' %
                  (i, len(test_dataset)/batch_size))

            input = input.cuda().float()
            output = model(input)
            pred = output.topk(1)[-1]

            all_idxs.append(labels)
            all_preds.append(pred.data.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy().astype('int64')
        all_idxs = torch.cat(all_idxs, dim=0).numpy(
        ).reshape(-1, 1).astype('int64')
        res = np.concatenate((all_idxs, all_preds), axis=1)

        # complement the missing data
        f1 = open(test_whole_file, 'r')
        with open(new_csv, 'w') as f3:
            new_csv_writer = csv.writer(f3, delimiter=',')
            new_csv_writer.writerow(['id', 'predicted'])

            for data in res:
                new_csv_writer.writerow(data)

            test_ids = f1.readlines()
            for idx in test_ids:
                idx = int(idx.strip())
                if idx not in all_idxs:
                    new_csv_writer.writerow(
                        [idx, np.random.randint(low=1, high=num_classes+1)])
        f1.close()


def avg_ensamble(preds_list, test_whole_file, new_csv):
    """ average ensamble strategy
    """

    for i, pred in enumerate(preds_list):
        arr = np.load(pred)
        if(i == 0):
            idxs = np.array(arr[:, 0], dtype='int32').reshape((-1, 1))
            res = np.array(arr[:, 1:], dtype='float32')
        else:
            res += arr[:, 1:]

    res /= (i+1)
    labels = (np.argmax(res, axis=1)+1).reshape((-1, 1))
    res = np.concatenate((idxs, labels), axis=1)

    # complement the missing data
    print(f'writing avg ensamble submission csv file')
    f1 = open(test_whole_file, 'r')
    with open(new_csv, 'w') as f3:
        new_csv_writer = csv.writer(f3, delimiter=',')
        new_csv_writer.writerow(['id', 'predicted'])

        for data in res:
            new_csv_writer.writerow(data)

        test_ids = f1.readlines()
        for idx in test_ids:
            idx = int(idx.strip())
            if idx not in idxs:
                new_csv_writer.writerow(
                    [idx, np.random.randint(low=1, high=num_classes+1)])
    f1.close()
    print('done')


def main():
    # train_ensamble()
    # weighted_ensamble(preds_list=p_list,
                      # test_whole_file=test_whole_file, new_csv=final_preds_csv)
    avg_ensamble(preds_list=p_list, test_whole_file=test_whole_file,
                 new_csv=final_preds_csv)


if __name__ == "__main__":
    main()
