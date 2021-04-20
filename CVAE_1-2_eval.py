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
import copy

def train_syn_val(config,im_net,loss_net,val_loader,syn_dataloader,optimizer,n_class):
    criterion=torch.nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    loss_net.train()
    im_net.eval()
    top1 = AverageMeter()
    #step=15
    best_acc=0
    best_epoch=0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1, last_epoch=-1)
    for epoch in range(20):
        for batch_num, batch_data in enumerate(syn_dataloader):
            feat, label = batch_data
            feat, label = feat.cuda(non_blocking=True), label.long().cuda(non_blocking=True)
            optimizer.zero_grad()
            output = loss_net(feat)
            prec1, class_acc, class_cnt, pred_prob= accuracy(output, label,n_class)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), feat.size(0))
            top1.update(prec1,feat.size(0))
        # scheduler make it better
        scheduler.step()

        val_acc, val_top1, pred_prob ,_,_= test(config, im_net, loss_net, val_loader, True, 1,eval_mode=1)
        if val_acc>best_acc:
            best_acc=val_acc
            best_epoch=epoch
            print('ACA: best valid {:.3f},  best epoches {} , current ACA pred meas {:.3f}, CLS {:.3f}'
                  .format(best_acc, best_epoch, val_acc, val_top1))

            best_model=copy.deepcopy(loss_net)

        if epoch % 5 == 0 and epoch != 0:
            print('%d epoch, train loss %.4f,top1 %.4f,' % (epoch, losses.avg,top1.avg))
    # best_model = copy.deepcopy(loss_net)
    return best_model


def test(config, im_net, loss_net, val_loader,CUDA,task_id,eval_mode=0,print_sign=0):
    batch_time = AverageMeter()
    top1 = AverageMeter()

    if task_id==0:
        n_lbl =config['n_train_lbl']
    else:
        n_lbl = config['n_test_lbl']
    true_labels=[]
    pred_labels=[]
    class_avg = ClassAverageMeter(n_lbl)
    im_net.eval()
    loss_net.eval()
    with torch.no_grad():
        for valid_batch_num, valid_batch_data in enumerate(val_loader):
            end = time.time()
            img, label = valid_batch_data
            # if task_id>0:
            #     label=label+150
            if CUDA:
                img, label = img.cuda(non_blocking=True), label.long().cuda(non_blocking=True)
            m = img.size(0)
            img_feat = im_net(img).reshape(m,-1)
            # img_feat=torch.flatten(img_feat,1)
            output = loss_net(img_feat)
            _, pred = output.topk(1, 1, True, True)
            prec1, class_acc, class_cnt, prec_prob = accuracy(output, label, n_lbl,bool_gzsl=eval_mode)
            true_labels.append(label)
            pred_labels.append(pred)
            top1.update(prec1, m)
            class_avg.update(class_acc, class_cnt, prec_prob)

            batch_time.update(time.time() - end)

            if print_sign and valid_batch_num % config['print_freq'] == 0:
                print('Test: [{0}/{1}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                      'Class avg {class_avg.avg:.3f} '.format(
                    valid_batch_num, len(val_loader), batch_time=batch_time,
                    class_avg=class_avg, top1=top1))
    if print_sign:
        print(class_avg.avg)
    return class_avg.avg, top1.avg, class_avg.pred_prob,torch.cat(true_labels),torch.cat(pred_labels).reshape(-1)

def main():
    CUDA = False
    if torch.cuda.is_available():
        CUDA=True
        print('cuda available')
        torch.backends.cudnn.benchmark = True
    config = config_process(parser.parse_args())
    print(config)

    with open('pkl/task_1_train.pkl', 'rb') as f:
        task_1_train = pkl.load(f)
    # with open('pkl/task_1_test.pkl', 'rb') as g:
    #     task_1_test = pkl.load(g)
    ######################## task_1_testval.pkl
    with open('pkl/task_1_test.pkl', 'rb') as g:
        task_1_testval = pkl.load(g)

    task_1_test=task_1_testval[:500]
    task_1_val=task_1_testval[500:]
    ###################################### task 1  test + val
    ###### task 0:seen training data and unseen test data
    examples, labels, class_map = image_load(config['class_file'], config['image_label'])
    ###### task 0: seen test data
    examples_0, labels_0, class_map_0 = image_load(config['class_file'], config['test_seen_classes'])

    datasets = split_byclass(config, examples, labels, np.loadtxt(config['attributes_file']), class_map)
    datasets_0 = split_byclass(config, examples_0, labels_0, np.loadtxt(config['attributes_file']), class_map)
    print('load the task 0 train: {} the task 1 as test: {}'.format(len(datasets[0][0]), len(datasets[0][1])))
    print('load task 0 test data {}'.format(len(datasets_0[0][0])))

    train_attr=F.normalize(datasets[0][3])
    test_attr=F.normalize(datasets[0][4])

    best_cfg = config
    best_cfg['n_classes'] = datasets[0][3].size(0)
    best_cfg['n_train_lbl'] = datasets[0][3].size(0)
    best_cfg['n_test_lbl'] = datasets[0][4].size(0)

    task_1_train_set = grab_data(best_cfg, task_1_train, datasets[0][2], True)
    task_1_test_set = grab_data(best_cfg, task_1_test, datasets[0][2], False)
    task_1_val_set = grab_data(best_cfg, task_1_val, datasets[0][2], False)
    task_0_seen_test_set = grab_data(best_cfg, datasets_0[0][0], datasets_0[0][2], False)

    base_model = models.__dict__[config['arch']](pretrained=True)
    if config['arch'].startswith('resnet'):
        FE_model = nn.Sequential(*list(base_model.children())[:-1])
    else:
        print('untested')
        raise NotImplementedError

    print('load pretrained FE_model')
    #######3 task id 'softmax'
    FE_path='./ckpts/{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
        config['dataset'],config['softmax_method'],config['arch'],
        config['task_id'],config['finetune'],'checkpoint.pth')

    FE_model.load_state_dict(torch.load(FE_path)['state_dict_FE'])
    for name, para in FE_model.named_parameters():
        para.requires_grad = False

    vae = VAE(
        encoder_layer_sizes=config['encoder_layer_sizes'],
        latent_size=config['latent_size'],
        decoder_layer_sizes=config['decoder_layer_sizes'],
        num_labels=config['num_labels'])

    vae_path='./ckpts/{}_{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
                config['dataset'], config['method'],config['softmax_method'],
                config['arch'], config['task_id'], config['finetune'], 'ckpt.pth')
    vae.load_state_dict(torch.load(vae_path))
    for name, para in vae.named_parameters():
        para.requires_grad = False

    FE_model.eval()
    vae.eval()
    # print(vae)
    if CUDA:
        FE_model=FE_model.cuda()
        vae=vae.cuda()

    #seen
    task_1_real_train=get_prev_feat(FE_model, task_1_train_set, CUDA)
    # task_0_real_val = get_prev_feat(FE_model, task_0_val_set, CUDA)

    print('have got real trainval feats and labels')
    print('...GENERATING fake features...')
    task_0_fake = generate_syn_feature(150, vae, train_attr, config['syn_num'], config)

    train_X = torch.cat((task_0_fake[0].cuda(), task_1_real_train[0].cuda()))
    train_Y = torch.cat((task_0_fake[1].cuda(), task_1_real_train[1].cuda() + 150))
    # train_X = task_0_fake[0].cuda()
    # train_Y = task_0_fake[1].cuda()
    test_Dataset = PURE(train_X, train_Y)

    test_dataloader = torch.utils.data.DataLoader(test_Dataset,
                                                 batch_size=256,
                                                 shuffle=True)

    test_loss_net = nn.Linear(in_features=2048, out_features=200).cuda()
    test_loss_net_optimizer = torch.optim.Adam(test_loss_net.parameters())

    print('...TRAIN test set CLASSIFIER...')
    # train_syn_val(test_loss_net,task_0_val_set, test_dataloader, test_loss_net_optimizer, 200)
    best_loss_net=train_syn_val(config, FE_model, test_loss_net, task_1_val_set, test_dataloader,
                                test_loss_net_optimizer, 200)
    test_loss_net=copy.deepcopy(best_loss_net)
    print('\n...TESTING... GZSL: 200 labels')

    # test_loss_net = torch.load('hist_files/vae_time_2_doulbe_distill_loss_net.pth')
    # torch.save(test_loss_net,'vae_time_2_loss_net.pth')
    test_0_acc, test_0_top1, _,true_labels_0,pred_labels_0 = test(config, FE_model, test_loss_net, task_0_seen_test_set,
                                      CUDA,0,eval_mode=0,print_sign=1)
    test_1_acc, test_1_top1, _, true_labels_1,pred_labels_1 = test(config, FE_model, test_loss_net, task_1_test_set,
                                      CUDA,1,eval_mode=1,print_sign=1)
    H=2*test_0_acc*test_1_acc/(test_0_acc+test_1_acc)
    print(H)
    OM=(3*test_0_acc+test_1_acc)/4
    print(OM)
    if not os.path.exists('results'):
        os.makedirs('results')

    file_name = '{}_{}_{}_{}.txt'.format(config['dataset'],config['arch'],config['method'], config['task_id'])
    with open('results/' + file_name, 'a') as fp:
        print(best_cfg, file=fp)
        print('task B: {:.3f}, task A:{:.3f}, H= {:.3f}, OM= {:.3f}  \n'.
              format(test_0_acc, test_1_acc,H,OM), file=fp)
    # np.save('confusion_matrix/vae_time_2_true_labels.npy',
    #         torch.cat((true_labels_0, true_labels_1 + 150)).detach().cpu().numpy())
    #
    # np.save('confusion_matrix/vae_time_2_pred_labels.npy',
    #         torch.cat((pred_labels_0, pred_labels_1)).detach().cpu().numpy())



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
    ############## task id
    parser.add_argument('--task_id', type=int, default=1,
                        help='task_id')
    parser.add_argument('--method', type=str, default='vae',
                        help='method')
    parser.add_argument('--step', default=15, type=int, metavar='NUM',
                        help='for SGD the default lr reducing step')
    parser.add_argument('--batch_size', default=64, type=int, metavar='NUM',
                        help='batch_size')

    parser.add_argument('--softmax_method', type=str, default='softmax',
                        help='softmax_method')
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

    # test_loss_net=torch.load('vae_time_2_loss_net.pth')
    # cls_1_weight=torch.norm(test_loss_net.weight[:150,:],dim=1).cpu().detach().numpy()
    # cls_2_weight = torch.norm(test_loss_net.weight[150:,:], dim=1).cpu().detach().numpy()
    #
    # cls_1_bias=test_loss_net.bias[:150].cpu().detach().numpy()
    # cls_2_bias=test_loss_net.bias[150:].cpu().detach().numpy()
    #
    # # sns.distplot(cls_1_weight, bins=150,label='task_A', color='r')
    # # sns.distplot(cls_2_weight,  bins=50,label='task_B', color='b')
    # sns.distplot(cls_1_bias, bins=150, label='task_A', color='r')
    # sns.distplot(cls_2_bias, bins=50, label='task_B', color='b')
    # plt.legend(loc='upper right', prop={'size': 18})
    # plt.show()
    # # print()