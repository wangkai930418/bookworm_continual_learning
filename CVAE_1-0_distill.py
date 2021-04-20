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
import copy


def test(config, im_net, loss_net, val_loader,CUDA):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    ########### n_train_lbl
    class_avg = ClassAverageMeter(config['n_test_lbl'])
    im_net.eval()
    loss_net.eval()
    with torch.no_grad():
        for valid_batch_num, valid_batch_data in enumerate(val_loader):
            end = time.time()
            img, label = valid_batch_data
            if CUDA:
                img, label = img.cuda(non_blocking=True), label.long().cuda(non_blocking=True)
            m = img.size(0)
            img_feat = im_net(img).reshape(m,-1)
            # img_feat=torch.flatten(img_feat,1)
            output = loss_net(img_feat)
            prec1, class_acc, class_cnt, prec_prob = accuracy(output, label, config['n_test_lbl'])

            top1.update(prec1, m)
            class_avg.update(class_acc, class_cnt, prec_prob)

            batch_time.update(time.time() - end)

            if valid_batch_num % config['print_freq'] == 0:
                print('Test: [{0}/{1}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                      'Class avg {class_avg.avg:.3f} '.format(
                    valid_batch_num, len(val_loader), batch_time=batch_time,
                    class_avg=class_avg, top1=top1))
    print(class_avg.avg)
    return class_avg.avg, top1.avg, class_avg.pred_prob


def train(config, im_net,im_net_prev, loss_net, train_loader, optimizer, criterion, epoch, CUDA):

    loss_net.train()
    if config['finetune']:
        im_net.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(config['n_test_lbl'])
    losses = AverageMeter()
    reg_losses = AverageMeter()
    dis_losses = AverageMeter()
    end = time.time()

    for batch_num, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, label = batch_data

        if CUDA:
            inputs, label = inputs.cuda(non_blocking=True), label.long().cuda(non_blocking=True)

        m = inputs.size(0)

        img_feat = im_net(inputs).reshape(m,-1)
        img_feat_prev = im_net_prev(inputs).reshape(m,-1)
        distill_loss = torch.sqrt(torch.sum((img_feat_prev - img_feat) ** 2, dim=1)).mean()
        # img_feat =torch.flatten(img_feat, 1)
        output = loss_net(img_feat)

        prec1, class_acc, class_cnt, pred_prob = accuracy(output, label, config['n_test_lbl'])
        reg_loss = criterion(output, label)
        loss=reg_loss+distill_loss

        reg_losses.update(reg_loss, m)
        dis_losses.update(distill_loss, m)
        losses.update(loss, m)
        top1.update(prec1, m)
        class_avg.update(class_acc, class_cnt, pred_prob)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_num % config['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                  'Reg_Loss {reg_loss.val:.4f} (avg: {reg_loss.avg:.4f}) '
                  'Dis_Loss {dis_loss.val:.4f} (avg: {dis_loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                  'Class avg {lbl_avg.avg:.3f} '.format(
                epoch, batch_num, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,reg_loss=reg_losses,dis_loss=dis_losses,
                lbl_avg=class_avg, top1=top1))


def main():
    CUDA = False
    if torch.cuda.is_available():
        CUDA=True
        print('cuda available')
        torch.backends.cudnn.benchmark = True
    config = config_process(parser.parse_args())
    print(config)
    ###### task 0:seen training data and unseen test data
    examples, labels, class_map = image_load(config['class_file'], config['image_label'])
    ###### task 0: seen test data
    examples_0, labels_0, class_map_0 = image_load(config['class_file'], config['test_seen_classes'])

    datasets = split_byclass(config, examples, labels, np.loadtxt(config['attributes_file']), class_map)

    with open('pkl/task_1_train.pkl', 'rb') as f:
        task_1_train = pkl.load(f)
    with open('pkl/task_1_test.pkl', 'rb') as g:
        task_1_testval = pkl.load(g)

    task_1_test=task_1_testval[:500]
    task_1_val=task_1_testval[500:]

    datasets_0 = split_byclass(config, examples_0, labels_0, np.loadtxt(config['attributes_file']), class_map)

    print('load the task 0 train: {} the task 1 as test: {}'.format(len(datasets[0][0]), len(datasets[0][1])))
    print('load task 0 test data {}'.format(len(datasets_0[0][0])))

    best_cfg = config
    best_cfg['n_classes'] = datasets[0][3].size(0)
    best_cfg['n_train_lbl'] = datasets[0][3].size(0)
    best_cfg['n_test_lbl'] = datasets[0][4].size(0)

    task_1_train_set = grab_data(best_cfg, task_1_train, datasets[0][2], True)
    task_1_test_set = grab_data(best_cfg, task_1_test, datasets[0][2], False)
    task_1_val_set = grab_data(best_cfg, task_1_val, datasets[0][2], False)
    # task_0_seen_train_set = grab_data(best_cfg, datasets[0][0], datasets[0][2], True)
    task_0_seen_test_set = grab_data(best_cfg, datasets_0[0][0], datasets_0[0][2], False)

    base_model = models.__dict__[config['arch']](pretrained=True)
    if config['arch'].startswith('resnet'):
        FE_model = nn.Sequential(*list(base_model.children())[:-1])
    else:
        print('untested')
        raise NotImplementedError

    print('load pretrained FE_model')
    FE_path = './ckpts/{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
        config['dataset'], 'softmax', config['arch'], 0, config['finetune'], 'checkpoint.pth')

    FE_model.load_state_dict(torch.load(FE_path)['state_dict_FE'])
    FE_model_prev=copy.deepcopy(FE_model)
    FE_model_prev.eval()
    for name, para in FE_model_prev.named_parameters():
        para.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss().cuda()
    classifier = nn.Linear(in_features=2048, out_features=50)

    if CUDA:
        FE_model=FE_model.cuda()
        FE_model_prev = FE_model_prev.cuda()
        classifier=classifier.cuda()

    if config['finetune']:
        FE_model.train()
        for name, para in FE_model.named_parameters():
            para.requires_grad = True

        optimizer = torch.optim.Adam([
            {'params': FE_model.parameters(), 'lr': config['ft_lr']},
            {'params': classifier.parameters(), 'lr': config['lr']}], )

    else:
        FE_model.eval()
        for name, para in FE_model.named_parameters():
            para.requires_grad = False

        optimizer = torch.optim.Adam([
            {'params': classifier.parameters(), 'lr': config['lr']}],)

    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,config['step'],gamma=0.1,last_epoch=-1)
    best_epoch = -1
    #per class
    best_acc = -1
    #total task
    best_top1 = -1

    for epoch in range(config['epoch']):
        print_learning_rate(optimizer)
        print('\nTRAIN ... ')
        # adjust_learning_rate(optimizer, epoch, config)
        train(config, FE_model, FE_model_prev,classifier, task_1_train_set, optimizer, criterion, epoch, CUDA)
        print('\nVAL ... ')
        val_acc, val_top1, pred_prob=test(config, FE_model, classifier, task_1_val_set,CUDA)

        is_best = val_acc > best_acc

        if is_best:
            best_epoch = epoch
            best_acc = val_acc
            best_top1=val_top1
            print('\n TEST...')
            test_acc, test_top1, pred_prob = test(config, FE_model, classifier, task_1_test_set, CUDA)

            print('\n ACA: best test {:.3f}, CLS:{:.3f} best epoches {}'
                  .format(test_acc, test_top1, best_epoch))

            save_model(config, FE_model, classifier,optimizer, epoch, best_acc, best_epoch, is_best,
                       './ckpts/{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
                        config['dataset'],config['method'],config['arch'],
                           config['task_id'],config['finetune'],'checkpoint.pth'))

        print('ACA: best valid {:.3f}, CLS:{:.3f} best epoches {} , current ACA pred meas {:.3f}, CLS {:.3f}'
              .format(best_acc, best_top1,best_epoch, val_acc,val_top1))

        scheduler.step()

    print('ACA: best valid {:.3f}, CLS:{:.3f} best epoches {}'.format(best_acc, best_top1, best_epoch))

    if not os.path.exists('results'):
        os.makedirs('results')

    file_name = '{}_{}_{}_{}.txt'.format(config['dataset'],config['arch'],config['method'], config['task_id'])
    with open('results/' + file_name, 'a') as fp:
        print(best_cfg, file=fp)
        print('best epoch = {}%'.format(best_epoch), file=fp)
        print('best vali: {}:PS ACA = {:.3f}%, CLS= {:.3f}'.format(config['dataset'], best_acc, best_top1), file=fp)
        print('best test: {:.3f}, CLS:{:.3f}'.format(test_acc, test_top1), file=fp)


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
    parser.add_argument('--model_root', default='./ckpts', metavar='DIRECTORY',
                        help='dataset to model directory')
    parser.add_argument('--result_root', default='./results', metavar='DIRECTORY',
                        help='dataset to result directory')
    parser.add_argument('--final_epoch', default=50, type=int, metavar='NUM',
                        help='last number of epoch')
    parser.add_argument('--finetune',action='store_true',
                        help='finetune awareness')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--ft_lr', '--finetuning-learning-rate', default=0.0001, type=float,
                        metavar='FT_LR', help='initial ft_lr learning rate')
    parser.add_argument('--epoch', default=50, type=int, metavar='NUM',
                        help='last number of epoch')
    parser.add_argument('--print_freq', default=10, type=int, metavar='NUM',
                        help='print frequence')
    ##############3333
    parser.add_argument('--task_id', type=int, default=1,
                        help='task_id')
    parser.add_argument('--method', type=str, default='softmax_distill',
                        help='method')
    parser.add_argument('--step', default=15, type=int, metavar='NUM',
                        help='for SGD the default lr reducing step')
    parser.add_argument('--batch_size', default=64, type=int, metavar='NUM',
                        help='batch_size')
    main()