import os
from data import guipang
import torch.utils.data as data
import copy
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
# from config import get_train_config
from data import guipang
from data import qiafan
# from models.MVHNet import *
# from utils import append_feature, calculate_map
import time
import shutil
from tensorboardX import SummaryWriter
import pdb
dataset_guipang= '/repository/gong/qiafan/guipangdata/'
# config
cfg = {
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'batch_size': 4,
    'max_epoch': 250,
    'checkpoint': 100,
    'milestones': [60, 100],
    'gamma': 0.1,
    'dataset_guipang': '/repository/gong/qiafan/guipangdata/',
    'dataset_qiafan': '/repository/gong/qiafan/dataset/',
    'cuda_devices': '6,7',
    'ckpt_root': '/repository/gong/qiafan/'
}
# print(os.path.join(dataset_guipang,'train')
# for filename in os.listdir(os.path.join(dataset_guipang,'train')):
#     if os.path.splitext(filename)[1]=='.jpg':
#         print(filename)
# images=[]
# annotations=[]

# for filename in os.listdir(os.path.join(dataset_guipang,'train')):
#     if os.path.splitext(filename)[1]=='.jpg':
#         images.append(filename)
#         annotations.append(os.path.splitext(filename)[0]+'.xml')

    # elif os.path.splitext(filename)[1]=='.xml':
    #     annotations.append(filename)
data_set = {
    x: guipang(cfg=cfg['dataset_guipang'], part=x) for x in ['train', 'val']
}
# data_set = {
#     x: dataset(cfg=cfg['dataset_qiafan'], part=x) for x in ['train', 'val']
# }
data_loader = {
    x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'],
                        num_workers=4, shuffle=True, pin_memory=False)
    for x in ['train', 'val']
}
# a=[1,2]
# print(a)
# pdb.set_trace()

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']


model = torchvision.models.detection.maskrcnn_resnet50_fpn()
model.cuda()
model = nn.DataParallel(model)
model.train()

for i, (images, annotations) in enumerate(data_loader['train']):
    # print(images)
    # # print(annotations['annotation']['object']['name'])
    # # annotations=torch.FloatTensor(list(map(int,annotations['annotation']['object']['name'])))
    # # print(annotations['annotation']['object']['bndbox']['xmin'])
    # print(annotations['annotation']['object']['bndbox'])

    # print(torch.tensor([list(map(int,annotations['annotation']['object']['bndbox']['xmin'])),
    #     list(map(int,annotations['annotation']['object']['bndbox']['ymin']))]).t())
    # pdb.set_trace()
    images = torch.cuda.FloatTensor(images.cuda())
    boxes = torch.cuda.FloatTensor([list(map(int,annotations['annotation']['object']['bndbox']['xmin'])),
                                    list(map(int,annotations['annotation']['object']['bndbox']['ymin'])),
                                    list(map(int,annotations['annotation']['object']['bndbox']['xmax'])),
                                    list(map(int,annotations['annotation']['object']['bndbox']['ymax']))]).t()
    labels = torch.cuda.LongTensor(
        list(map(int, annotations['annotation']['object']['name'])))

    with torch.set_grad_enabled(phrase == 'train'):
        outputs = model(images,boxes=boxes,labels=labels)
        print(outputs)
        pdb.set_trace()
