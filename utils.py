import numpy as np
import torch
import math
import os
import random
import torch.backends.cudnn as cudnn
import shutil
import torchvision.transforms as transforms
from torch.utils import data
from dataset.dataset import DataSet,inst_DataSet

def accuracy(output_vec, target, n_labels,bool_gzsl=0):
    """Computes the precision@k for the specified values of k"""
    output = output_vec

    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    class_accuracy = torch.zeros(n_labels)
    class_cnt = torch.zeros(n_labels)
    #torch.sum(pred.reshape(-1)==target)
    prec = 0.0
    pred_prob = []
    for i in range(batch_size):
        t = target[i]
        pred_prob.append(output[i][t])
        aim = t
        if bool_gzsl:
            # 150 need to be changed
            aim=aim+150
        if pred[i] == aim:
            prec += 1
            class_accuracy[t] += 1
        class_cnt[t] += 1
    return prec * 100.0 / batch_size, class_accuracy, class_cnt, pred_prob

def train_syn(loss_net,syn_dataloader,optimizer,n_class):
    criterion=torch.nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    loss_net.train()
    top1 = AverageMeter()
    #step=15
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1, last_epoch=-1)
    for epoch in range(50):
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

        if epoch % 10 == 0 and epoch != 0:
            print('%d epoch, train loss %.4f,top1 %.4f,' % (epoch, losses.avg,top1.avg))

def image_load(class_file, label_file):
    with open(class_file, 'r') as f:
        class_names = [l.strip() for l in f.readlines()]
    class_map = {}
    for i,l in enumerate(class_names):
        items = l.split()
        class_map[items[-1]] = i
    #print(class_map)
    examples = []
    labels = {}
    with open(label_file, 'r') as f:
        image_label = [l.strip() for l in f.readlines()]
    for lines in image_label:
        items = lines.split()
        examples.append(items[0])
        labels[items[0]] = int(items[1])
    return examples,labels, class_map

def split_byclass(config, examples,labels, attributes, class_map):
    with open(config['train_classes'], 'r') as f:
        train_lines = [l.strip() for l in f.readlines()]
    with open(config['test_classes'], 'r') as f:
        test_lines = [l.strip() for l in f.readlines()]
    train_attr = []
    test_attr = []
    train_class_set = {}
    for i,name in enumerate(train_lines):
        idx = class_map[name]
        train_class_set[idx] = i
        # idx is its real label
        train_attr.append(attributes[idx])
    test_class_set = {}
    for i,name in enumerate(test_lines):
        idx = class_map[name]
        test_class_set[idx] = i
        test_attr.append(attributes[idx])
    train = []
    test = []
    label_map = {}
    for ins in examples:
        v = labels[ins]
        # inital label
        if v in train_class_set:
            train.append(ins)
            label_map[ins] = train_class_set[v]
        else:
            test.append(ins)
            label_map[ins] = test_class_set[v]
    train_attr = torch.from_numpy(np.array(train_attr,dtype='float')).float()
    test_attr = torch.from_numpy(np.array(test_attr,dtype='float')).float()
    return [(train,test,label_map,train_attr,test_attr)]

def grab_data(config, examples, labels, is_train=True):
    params = {'batch_size': config['batch_size'],
              'num_workers': 4,
              'pin_memory': True,
              'sampler': None}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tr_transforms, ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.08, 1), (0.5, 4.0 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if is_train:
        params['shuffle'] = True
        params['sampler'] = None
        data_set = data.DataLoader(DataSet(config, examples, labels, tr_transforms, is_train), **params)
    else:
        params['shuffle'] = False
        data_set = data.DataLoader(DataSet(config, examples, labels, ts_transforms, is_train), **params)
    return data_set

def inst_grab_data(config, examples, labels,name_attr_lbl_dict, is_train=True):
    params = {'batch_size': config['batch_size'],
              'num_workers': 4,
              'pin_memory': True,
              'sampler': None}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tr_transforms, ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.08, 1), (0.5, 4.0 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if is_train:
        params['shuffle'] = True
        params['sampler'] = None
        data_set = data.DataLoader(inst_DataSet(config, examples, labels,
                                                name_attr_lbl_dict,tr_transforms, is_train), **params)
    else:
        params['shuffle'] = False
        data_set = data.DataLoader(inst_DataSet(config, examples, labels,
                                                name_attr_lbl_dict, ts_transforms, is_train), **params)
    return data_set


def config_process(config):
    if config.dataset == 'CUB200':
        config.imagedir='/home/wangkai/LLL/CUB/CUB_200_2011/CUB_200_2011'
        config.image_dir = os.path.join(config.imagedir, 'images/')
        config.txtdir='CUB'
        config.class_file = os.path.join(config.data_root, config.txtdir, 'classes.txt')
        config.image_label = os.path.join(config.data_root, config.txtdir, 'image_label_PS.txt')
        config.attributes_file = os.path.join(config.data_root, config.txtdir, 'class_attributes.txt')
        config.train_classes = os.path.join(config.data_root, config.txtdir, 'trainvalclasses.txt')
        config.test_classes = os.path.join(config.data_root, config.txtdir, 'testclasses.txt')
        config.test_seen_classes = os.path.join(config.data_root, config.txtdir, 'test_seen_PS.txt')
    elif config.dataset=='AWA2':
        config.imagedir = '/home/wangkai/LLL/AWA2/Animals_with_Attributes2'
        config.image_dir = os.path.join(config.imagedir, 'JPEGImages/')
        config.txtdir = 'AWA2'
        config.class_file = os.path.join(config.data_root, config.txtdir, 'classes.txt')
        config.image_label = os.path.join(config.data_root, config.txtdir, 'image_label_PS.txt')
        config.attributes_file = os.path.join(config.data_root, config.txtdir, 'predicate-matrix-continuous.txt')
        config.train_classes = os.path.join(config.data_root, config.txtdir, 'trainvalclasses.txt')
        config.test_classes = os.path.join(config.data_root, config.txtdir, 'testclasses.txt')
        config.test_seen_classes = os.path.join(config.data_root, config.txtdir, 'test_seen_PS.txt')

    if not os.path.exists(config.result_root):
        os.makedirs(config.result_root)
    if not os.path.exists(config.model_root):
        os.makedirs(config.model_root)
    # namespace ==> dictionary
    return vars(config)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ClassAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, n_labels):
        self.reset(n_labels)

    def reset(self, n_labels):
        self.n_labels = n_labels
        self.acc = torch.zeros(n_labels)
        self.cnt = torch.Tensor([1e-8] * n_labels)
        self.pred_prob = []
        self.avg=0

    def update(self, val, cnt, pred_prob):
        self.acc += val
        self.cnt += cnt
        self.avg = 100 * self.acc.dot(1.0 / self.cnt).item() / self.n_labels
        self.pred_prob += pred_prob
        # print ('pred',len(self.pred_prob))

def save_checkpoint(config, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './ckpts/{}_{}_{}_task_id_{}_finetune_{}_model_best.pth'.format(
            config['dataset'],config['method'],config['arch'], config['task_id'],config['finetune']))

def save_model(config, model,classifier, optimizer, epoch, best_meas, best_epoch, is_best, fname):
    save_checkpoint(config, {
        'epoch': epoch + 1,
        'arch': config['arch'],
        'state_dict_FE': model.state_dict(),
        'state_dict_CLS':classifier.state_dict(),
        'best_meas': best_meas,
        'best_epoch': best_epoch,
        'optimizer': optimizer.state_dict(),
    }, is_best, fname)

def print_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        print('current step learning rate {}'.format(param_group['lr']))

def loss_fn(recon_x, x, mean, log_var):
    BCE=torch.sum((recon_x-x)**2)/x.size(1)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/mean.size(1)
    #beta vae
    return (BCE + KLD*0.1) / x.size(0)

def get_prev_feat(im_net,train_loader,CUDA):
    seen_feat = []
    seen_label = []
    for batch_num, batch_data in enumerate(train_loader):
        img, label = batch_data
        if CUDA:
            img, label = img.cuda(non_blocking=True), label.long().cuda(non_blocking=True)
        img_feat = im_net(img).squeeze()
        seen_feat.append(img_feat)
        seen_label.append(label)
    seen_feat = torch.cat(seen_feat)
    seen_label = torch.cat(seen_label)
    return seen_feat,seen_label

def generate_syn_feature(num_class,netG, attribute, num,args):
    #number of categories
    nclass = num_class
    ## num is number of fake features
    syn_feature = torch.FloatTensor(nclass * num, args['attSize'])
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, args['num_labels'])
    syn_att = syn_att.cuda()
    netG.eval()
    for i in range(nclass):
        iclass = i
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        # syn_noise.normal_(0, 1)
        # volatile removed
        output = netG.inference(num, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

def inst_generate_syn_feature(num_class,netG, attribute,label, num,args):
    #number of categories
    nclass = num_class
    ## num is number of fake features
    syn_feature = torch.FloatTensor(nclass * num, args['attSize'])
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, args['num_labels'])
    syn_att = syn_att.cuda()
    netG.eval()
    for i in range(nclass):
        iclass = i
        iclass_att = attribute[iclass]
        iclass_lbl=label[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        # syn_noise.normal_(0, 1)
        # volatile removed
        output = netG.inference(num, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass_lbl)

    return syn_feature, syn_label