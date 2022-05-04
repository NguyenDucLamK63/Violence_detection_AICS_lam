import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from epoch import train, val, test
from model import VioNet_C3D, VioNet_ConvLSTM, VioNet_densenet, VioNet_densenet_lean
from dataset import VioDB
from config import Config

from spatial_transforms import Compose, ToTensor, Normalize
from spatial_transforms import GroupRandomHorizontalFlip, GroupRandomScaleCenterCrop, GroupScaleCenterCrop
from temporal_transforms import CenterCrop, RandomCrop
from target_transforms import Label, Video
import matplotlib.pyplot as plt
from utils import Log

import time

def main(config,cv):
    # load model
    if config.model == 'c3d':
        model, params = VioNet_C3D(config)
    elif config.model == 'convlstm':
        model, params = VioNet_ConvLSTM(config)
    elif config.model == 'densenet':
        model, params = VioNet_densenet(config)
    elif config.model == 'densenet_lean':
        model, params = VioNet_densenet_lean(config)
    # default densenet
    else:
        model, params = VioNet_densenet_lean(config)

    # dataset
    dataset = config.dataset
    sample_size = config.sample_size
    stride = config.stride
    sample_duration = config.sample_duration

    # cross validation phase
    cv = config.num_cv

    # train set
    crop_method = GroupRandomScaleCenterCrop(size=sample_size)
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    spatial_transform = Compose(
        [crop_method,
         GroupRandomHorizontalFlip(),
         ToTensor(), norm])
    temporal_transform = RandomCrop(size=sample_duration, stride=stride)
    target_transform = Label()

    train_batch = config.train_batch

    train_data = VioDB('../VioDB/{}_jpg/'.format(dataset),
                       '../VioDB/{}_jpg{}.json'.format(dataset, cv), 'training',
                       spatial_transform, temporal_transform, target_transform)
    train_loader = DataLoader(train_data,
                              batch_size=train_batch,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)


    # val set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    temporal_transform = CenterCrop(size=sample_duration, stride=stride)
    target_transform = Label()

    val_batch = config.val_batch

    val_data = VioDB('../VioDB/{}_jpg/'.format(dataset),
                     '../VioDB/{}_jpg{}.json'.format(dataset, cv), 'validation',
                     spatial_transform, temporal_transform, target_transform)
    val_loader = DataLoader(val_data,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # make dir
    if not os.path.exists('/content/drive/MyDrive/Log_train_Violence/pth'):
        os.mkdir('/content/drive/MyDrive/Log_train_Violence/pth')
    if not os.path.exists('/content/drive/MyDrive/Log_train_Violence/log'):
        os.mkdir('/content/drive/MyDrive/Log_train_Violence/log')
    if not os.path.exists('/content/drive/MyDrive/Log_train_Violence/img'):
        os.mkdir('/content/drive/MyDrive/Log_train_Violence/img')

    # log
    batch_log = Log(
        '/content/drive/MyDrive/Log_train_Violence/log/{}_fps{}_{}_batch{}_lan1.log'.format(
            config.model,
            sample_duration,
            dataset,
            cv,
        ), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_log = Log(
        '/content/drive/MyDrive/Log_train_Violence/log/{}_fps{}_{}_epoch{}_lan1.log'.format(config.model, sample_duration,
                                               dataset, cv),
        ['epoch', 'loss', 'acc', 'lr'])
    val_log = Log(
        '/content/drive/MyDrive/Log_train_Violence/log/{}_fps{}_{}_val{}_lan1.log'.format(config.model, sample_duration,
                                             dataset, cv),
        ['epoch', 'loss', 'acc'])

    # prepare
    criterion = nn.CrossEntropyLoss().to(device)

    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay

    optimizer = torch.optim.SGD(params=params,
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)

    acc_baseline = config.acc_baseline
    loss_baseline = 1
    h_train_loss = []
    h_train_acc = []
    h_val_loss = []
    h_val_acc = []
    for i in range(config.num_epoch):
        start=time.time()
        train_loss, train_acc = train(i, train_loader, model, criterion, optimizer, device, batch_log,
              epoch_log)
        val_loss, val_acc = val(i, val_loader, model, criterion, device,
                                val_log)
        h_train_loss.append(train_loss)
        h_train_acc.append(train_acc)
        h_val_loss.append(val_loss)
        h_val_acc.append(val_acc)
        scheduler.step(val_loss)
        if val_acc > acc_baseline or (val_acc >= acc_baseline and
                                      val_loss < loss_baseline):
            torch.save(
                model.state_dict(),
                '/content/drive/MyDrive/Log_train_Violence/pth/{}_fps{}_{}{}_{}_{:.4f}_{:.6f}_lan1.pth'.format(
                    config.model, sample_duration, dataset, cv, i, val_acc,
                    val_loss))
            acc_baseline = val_acc
            loss_baseline = val_loss
        end=time.time()
        print(f"time: {end - start}")
        fig = plt.figure(figsize=(20,10))

        plt.title("Train - Validation Loss")
        plt.plot(h_train_loss, label='train')
        plt.plot(h_val_loss, label='validation')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend(loc='best')
        plt.savefig("/content/drive/MyDrive/Log_train_Violence/img/loss_densenet_lean_vif_-3_45_%s.png"%cv)

        fig = plt.figure(figsize=(20,10))
        plt.title("Train - Validation Accuracy")
        plt.plot(h_train_acc, label='train')
        plt.plot(h_val_acc, label='validation')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.legend(loc='best')
        plt.savefig("/content/drive/MyDrive/Log_train_Violence/img/acc_densenet_lean_vif_45_-3_%s.png"%cv)
        
    return h_train_acc, h_train_loss, h_val_acc, h_val_loss
    
    
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = Config(
        'densenet_lean',  # c3d, convlstm, densenet, densenet_lean
        'vif',
        device=device,
        num_epoch=45,
        acc_baseline=0.92,
        ft_begin_idx=0,
    )

    # train params for different datasets
    configs = {
        'hockey': {
            'lr': 1e-3,
            'batch_size': 32
        },
        'movie': {
            'lr': 1e-3,
            'batch_size': 32
        },
        'vif': {
            'lr': 1e-3,
            'batch_size': 32
        },
        'mix': {
            'lr': 1e-3,
            'batch_size': 32
        }
    }

    for dataset in ['vif']:
        config.dataset = dataset
        config.train_batch = configs[dataset]['batch_size']
        config.val_batch = configs[dataset]['batch_size']
        config.learning_rate = configs[dataset]['lr']
        #5 fold cross validation
        for cv in range(1, 6):
            config.num_cv = cv
            h_train_acc, h_train_loss, h_val_acc, h_val_loss = main(config,cv)
            
    fig = plt.figure(figsize=(20,10))
    plt.title("Train - Validation Loss")
    plt.plot(h_train_loss, label='train')
    plt.plot(h_val_loss, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    plt.savefig("/content/drive/MyDrive/Log_train_Violence/img/loss_densenet_lean_vif_-3_45_total.png")

    fig = plt.figure(figsize=(20,10))
    plt.title("Train - Validation Accuracy")
    plt.plot(h_train_acc, label='train')
    plt.plot(h_val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.savefig("/content/drive/MyDrive/Log_train_Violence/img/acc_densenet_lean_vif_-3_45_total.png")