# coding:utf-8
import random
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data
import numpy as np
import torchvision.transforms as transforms
import argparse
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import os
import time
import shutil
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from utils import *
import pickle as pkl
from model.vae import VAE
from dataset.dataset import PURE


def train(epoch,im_net, vae, train_loader, optimizer, criterion, train_attr,CUDA):
    losses=AverageMeter()
    vae.train()
    for batch_num, batch_data in enumerate(train_loader):
        img, label = batch_data
        label_embed=train_attr[label]
        if CUDA:
            img, label ,label_embed= img.cuda(non_blocking=True), \
                         label.long().cuda(non_blocking=True),label_embed.cuda(non_blocking=True)
        optimizer.zero_grad()
        img_feat = im_net(img).squeeze()
        recon_x, mean, log_var, z = vae(img_feat,label_embed)

        loss=criterion(recon_x, img_feat, mean, log_var)
        losses.update(loss.item(),img.size(0))
        loss.backward()
        optimizer.step()
        if batch_num % 10 == 0 and batch_num != 0:
            print('%d batch_num, loss:%.4f, loss_avg:%.4f' % (batch_num, loss, losses.avg))
    print('%d epoch, train loss %.4f \n'% (epoch, losses.avg))


def main():
    CUDA = False
    if torch.cuda.is_available():
        CUDA=True
        print('cuda available')
        torch.backends.cudnn.benchmark = True
    config = config_process(parser.parse_args())
    print(config)

    with open('pkl/task_0_train.pkl', 'rb') as f:
        task_0_train =pkl.load(f)

    ###### task 0:seen training data and unseen test data
    examples, labels, class_map = image_load(config['class_file'], config['image_label'])

    datasets = split_byclass(config, examples, labels, np.loadtxt(config['attributes_file']), class_map)
    print('load the task 0 train: {} the task 1 as test: {}'.format(len(datasets[0][0]), len(datasets[0][1])))

    train_attr=F.normalize(datasets[0][3])

    best_cfg = config
    best_cfg['n_classes'] = datasets[0][3].size(0)
    best_cfg['n_train_lbl'] = datasets[0][3].size(0)
    best_cfg['n_test_lbl'] = datasets[0][4].size(0)

    task_0_train_set=grab_data(best_cfg, task_0_train, datasets[0][2], True)

    base_model = models.__dict__[config['arch']](pretrained=False)
    if config['arch'].startswith('resnet'):
        FE_model = nn.Sequential(*list(base_model.children())[:-1])
    else:
        print('untested')
        raise NotImplementedError

    print('load pretrained FE_model')
    FE_path='./ckpts/{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
        config['dataset'],config['softmax_method'],config['arch'],config['task_id'],config['finetune'],'checkpoint.pth')

    FE_model.load_state_dict(torch.load(FE_path)['state_dict_FE'])
    for name, para in FE_model.named_parameters():
        para.requires_grad = False
    #####################################################
    FE_model.eval()
    vae = VAE(
        encoder_layer_sizes=config['encoder_layer_sizes'],
        latent_size=config['latent_size'],
        decoder_layer_sizes=config['decoder_layer_sizes'],
        num_labels=config['num_labels'])

    print(vae)
    if CUDA:
        FE_model=FE_model.cuda()
        vae=vae.cuda()

    optimizer = torch.optim.Adam(vae.parameters(), lr=config['lr'])

    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,config['step'],gamma=0.1,last_epoch=-1)
    criterion=loss_fn
    print('have got real trainval feats and labels')

    for epoch in range(config['epoch']):
        print('\n epoch: %d'%epoch)
        print('...TRAIN...')
        print_learning_rate(optimizer)
        train(epoch, FE_model, vae, task_0_train_set, optimizer, criterion, train_attr, CUDA)
        scheduler.step()

    # vae_ckpt_name='./ckpts/{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
    #     config['dataset'],config['method'],config['arch'],config['task_id'],config['finetune'],'ckpt.pth')
    # torch.save(vae.state_dict(), vae_ckpt_name)


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description=' source code')
    parser.add_argument('--dataset', default='CUB200', choices=['CUB200', 'AWA2'], metavar='NAME',
                        help='dataset name')
    parser.add_argument('--arch', default='resnet101', type=str, metavar='DIRECTORY',
                        help='name of arch')
    parser.add_argument('--data_root', default='./data', type=str, metavar='DIRECTORY',
                        help='path to data directory')
    parser.add_argument('--model_root', default='./models', metavar='DIRECTORY',
                        help='dataset to model directory')
    parser.add_argument('--result_root', default='./results', metavar='DIRECTORY',
                        help='dataset to result directory')
    parser.add_argument('--final_epoch', default=50, type=int, metavar='NUM',
                        help='last number of epoch')

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--ft_lr', '--finetuning-learning-rate', default=0.0001, type=float,
                        metavar='FT_LR', help='initial ft_lr learning rate')
    parser.add_argument('--epoch', default=50, type=int, metavar='NUM',
                        help='last number of epoch')
    parser.add_argument('--print_freq', default=10, type=int, metavar='NUM',
                        help='print frequence')
    parser.add_argument('--task_id', type=int, default=0,
                        help='task_id')
    ############################################# distill on first task
    parser.add_argument('--method', type=str, default='vae',
                        help='method')
    parser.add_argument('--softmax_method', type=str, default='softmax',
                        help='softmax_method')
    parser.add_argument('--step', default=15, type=int, metavar='NUM',
                        help='for SGD the default lr reducing step')
    parser.add_argument('--batch_size', default=64, type=int, metavar='NUM',
                        help='batch_size')

    ######### parameters need to select
    parser.add_argument('--finetune', action='store_true',
                        help='finetune awareness')
    ########## vae parameters
    parser.add_argument("--encoder_layer_sizes", type=int, nargs='+', default=[2048,512, 256])
    parser.add_argument("--decoder_layer_sizes", type=int, nargs='+', default=[1024, 2048])
    parser.add_argument("--latent_size", type=int, default=50)
    parser.add_argument("--attSize", type=int, default=2048)
    parser.add_argument("--num_labels", type=int, default=312)
    parser.add_argument('--syn_num', type=int, default=300,
                        help='early stop')
    main()