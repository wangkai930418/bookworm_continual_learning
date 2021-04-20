import torch
from torch.utils.data import Dataset
import pickle as pkl
from PIL import Image
import os
from PIL import Image
from torch.utils import data

class DataSet(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, args, examples, labels, transform, is_train):
        'Initialization'
        self.labels = labels
        self.examples = examples
        self.transform = transform
        self.image_dir = args['image_dir']
        self.args = args
        self.n_classes = self.args['n_classes']
        self.is_train = is_train

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def __getitem__(self, idx):
        'Generates one sample of data'
        id = self.examples[idx]
        # Convert to RGB to avoid png.
        X = Image.open(self.image_dir + id).convert('RGB')
        X = self.transform(X)
        label = self.labels[id]
        return X,label


class inst_DataSet(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, args, examples, labels, name_attr_lbl_dict,transform, is_train):
        'Initialization'
        self.labels = labels
        self.examples = examples
        self.transform = transform
        self.image_dir = args['image_dir']
        self.args = args
        self.n_classes = self.args['n_classes']
        self.is_train = is_train
        self.name_attr_lbl_dict=name_attr_lbl_dict

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def __getitem__(self, idx):
        'Generates one sample of data'
        id = self.examples[idx]
        # Convert to RGB to avoid png.
        X = Image.open(self.image_dir + id).convert('RGB')
        X = self.transform(X)
        label = self.labels[id]
        attr=self.name_attr_lbl_dict[id][0]
        return X,label,attr



class PURE(Dataset):
    def __init__(self,feat,label,transform=None):
        self.feat = feat
        self.label=label

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, index):
        return self.feat[index],self.label[index]