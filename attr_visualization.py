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
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.manifold import TSNE
import pickle as pkl
import matplotlib.cm as cm
import collections


def main():
    file_name='data/CUB/attributes.txt'
    f = open(file_name, 'r')
    content = f.readlines()
    atts = []
    for item in content:
        atts.append(item.strip().split(' ')[1])

    CUDA = False
    if torch.cuda.is_available():
        CUDA=True
        print('cuda available')
        torch.backends.cudnn.benchmark = True
    config = config_process(parser.parse_args())
    print(config)

    # pkl_name='./pkl/{}_{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
    #             config['dataset'], config['method'], config['softmax_method'],config['arch'],
    #             config['task_id'], config['finetune'], '.pkl')
    #
    # with open(pkl_name,'rb') as f:
    #     feat_dict=pkl.load(f)

    with open('pkl/task_0_train.pkl', 'rb') as f:
        task_0_train =pkl.load(f)

    with open('pkl/task_1_train.pkl', 'rb') as f:
        task_1_train = pkl.load(f)

    ###### task 0:seen training data and unseen test data
    examples, labels, class_map = image_load(config['class_file'], config['image_label'])
    ###### task 0: seen test data
    examples_0, labels_0, class_map_0 = image_load(config['class_file'], config['test_seen_classes'])

    datasets = split_byclass(config, examples, labels, np.loadtxt(config['attributes_file']), class_map)
    datasets_0 = split_byclass(config, examples_0, labels_0, np.loadtxt(config['attributes_file']), class_map)
    print('load the task 0 train: {} the task 1 as test: {}'.format(len(datasets[0][0]), len(datasets[0][1])))
    print('load task 0 test data {}'.format(len(datasets_0[0][0])))

    classes_text_embedding = torch.eye(312, dtype=torch.float32)
    test_attr = classes_text_embedding[:, :]

    train_attr=F.normalize(datasets[0][3])
    # test_attr=F.normalize(datasets[0][4])

    best_cfg = config
    best_cfg['n_classes'] = datasets[0][3].size(0)
    best_cfg['n_train_lbl'] = datasets[0][3].size(0)
    best_cfg['n_test_lbl'] = datasets[0][4].size(0)

    task_0_train_set=grab_data(best_cfg, task_0_train, datasets[0][2], True)
    task_1_train_set = grab_data(best_cfg, task_1_train, datasets[0][2], False)

    base_model = models.__dict__[config['arch']](pretrained=True)
    if config['arch'].startswith('resnet'):
        FE_model = nn.Sequential(*list(base_model.children())[:-1])
    else:
        print('untested')
        raise NotImplementedError

    print('load pretrained FE_model')
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
    vae2 = VAE(
        encoder_layer_sizes=config['encoder_layer_sizes'],
        latent_size=config['latent_size'],
        decoder_layer_sizes=config['decoder_layer_sizes'],
        num_labels=config['num_labels'])
    vae_path='./ckpts/{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
                config['dataset'], config['method'], config['arch'],
                config['task_id'], config['finetune'], 'ckpt.pth')

    vae_path = './ckpts/{}_{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
        config['dataset'], 'vae', 'softmax_distill', config['arch'],
        1, config['finetune'], 'ckpt.pth')

    vae2_path = './ckpts/{}_{}_{}_{}_task_id_{}_finetune_{}_{}'.format(
        config['dataset'],'vae_distill', 'softmax_distill', config['arch'],
        1, config['finetune'], 'ckpt.pth')

    vae.load_state_dict(torch.load(vae_path))
    vae2.load_state_dict(torch.load(vae2_path))
    for name, para in vae.named_parameters():
        para.requires_grad = False
    for name, para in vae2.named_parameters():
        para.requires_grad = False

    if CUDA:
        FE_model=FE_model.cuda()
        vae=vae.cuda()
        vae2=vae2.cuda()

    ATTR_NUM=312
    SYN_NUM=config['syn_num']
    attr_feat,attr_lbl=generate_syn_feature(ATTR_NUM, vae, test_attr, SYN_NUM, config)
    attr_feat2, attr_lbl2 = generate_syn_feature(ATTR_NUM, vae2, test_attr, SYN_NUM, config)

    with open('attr_tsne_data/attr_vae_time_2_fe_distill.pkl', 'wb') as f:
        pkl.dump(attr_feat,f)
    with open('attr_tsne_data/attr_vae_time_2_double_distill.pkl', 'wb') as g:
        pkl.dump(attr_feat2,g)

    colors = cm.rainbow(np.linspace(0, 1, ATTR_NUM))
    fig = plt.figure(figsize=(16,9))
    tsne = TSNE(n_components=2)

    feat = torch.cat((attr_feat, attr_feat2))
    tsne_results = tsne.fit_transform(feat)
    color_ind=colors[attr_lbl]
    ax = fig.add_subplot(1, 1, 1)

    for i in range(ATTR_NUM):
        ax.scatter(tsne_results[i*SYN_NUM:(i+1)*SYN_NUM, 0], tsne_results[i*SYN_NUM:(i+1)*SYN_NUM, 1],
                   label=atts[i],c=np.tile(colors[i].reshape(1, -1), (SYN_NUM, 1)), s=20, marker='X')

    result=tsne_results[ATTR_NUM*SYN_NUM:,:]
    for j in range(ATTR_NUM):
        ax.scatter(result[j * SYN_NUM:(j + 1) * SYN_NUM, 0], result[j * SYN_NUM:(j + 1) * SYN_NUM, 1],
                   label=atts[j], c=np.tile(colors[j].reshape(1, -1), (SYN_NUM, 1)), s=20, marker='o')
    # ax.scatter(tsne_results[:, 0], tsne_results[:, 1],c=color_ind, s=20, marker='X')
    plt.legend()
    plt.show()


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
    parser.add_argument('--syn_num', type=int, default=100,
                        help='early stop')
    main()