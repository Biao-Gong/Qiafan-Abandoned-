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


def train_guipang(model, criterion, optimizer, scheduler, cfg):

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']):
        for phrase in ['train', 'val']:

            nowepochiter = (epoch-1)*len(data_loader[phrase])
            if phrase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            ##################################################
            running_loss = 0.0
            running_corrects = 0
            ft_all, lbl_all = None, None
            ##################################################

            for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader[phrase]):

                optimizer.zero_grad()

                centers = torch.cuda.FloatTensor(centers.cuda())
                corners = torch.cuda.FloatTensor(corners.cuda())
                normals = torch.cuda.FloatTensor(normals.cuda())
                neighbor_index = torch.cuda.LongTensor(neighbor_index.cuda())
                targets = torch.cuda.LongTensor(targets.cuda())

                with torch.set_grad_enabled(phrase == 'train'):
                    fea_hash_mesh, fea_hash_sensib_mesh, outputs = model(
                        (centers, corners, normals, neighbor_index))
                    _, preds = torch.max(outputs, 1)
                    # loss = criterion['crossentropyloss'](
                    #     outputs, targets)+criterion[cfg['maskloss']](fea_hash_sensib_mesh, fea_hash_mesh)
                    lossa = criterion['crossentropyloss'](outputs, targets)
                    lossb = criterion[cfg['maskloss']](
                        fea_hash_sensib_mesh, fea_hash_mesh)
                    # loss=lossa+lossa*lossb
                    loss = lossa+lossb
                    if phrase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item() * centers.size(0)
                    running_corrects += torch.sum(preds == targets.data)

                if phrase == 'train':
                    writer.add_scalar('mesh/realtime/globalLoss',
                                      loss.item(), i+nowepochiter)

            epoch_loss = running_loss / len(data_set[phrase])
            epoch_acc = running_corrects.double() / len(data_set[phrase])

            if phrase == 'train':
                writer.add_scalar('mesh/epoch/globalLoss', epoch_loss, epoch)
                writer.add_scalar('mesh/epoch/Acc', epoch_acc, epoch)

            if phrase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.module.state_dict())
                    torch.save(best_model_wts, the_ckpt_root+'best.pkl')
                if epoch % cfg['checkpoint'] == 0:
                    torch.save(copy.deepcopy(model.module.state_dict()),
                               the_ckpt_root+'{}.pkl'.format(epoch))
                writer.add_scalar(
                    'mesh/epochtest/globalLoss', epoch_loss, epoch)
                writer.add_scalar('mesh/epochtest/Acc', epoch_acc, epoch)

    return best_model_wts


if __name__ == '__main__':

    ######################################
    # config
    cfg = {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'batch_size': 60,
        'max_epoch': 250,
        'checkpoint': 100,
        'milestones': [60, 100],
        'gamma': 0.1,
        'dataset_guipang': '/repository/gong/qiafan/guipangdata/',
        'dataset_qiafan': '/repository/gong/qiafan/dataset/',
        'cuda_devices': '6,7',
        'ckpt_root': '/repository/gong/qiafan/'
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
    time_TrainStart = str(int(time.time()))
    the_ckpt_root = cfg['ckpt_root']+time_TrainStart+'/'
    os.mkdir(the_ckpt_root)
    # tensorboardx
    writer = SummaryWriter('runs/'+time_TrainStart+'_'+str(os.getpid()))

    ######################################

    # dataset
    # data_set = {
    #     x: torchvision.datasets.VOCDetection(cfg['dataset_guipang'], year='2007', image_set=x, download=False, transform=None, target_transform=None, transforms=None)
    #     for x in ['train', 'val']
    # }
    data_set = {
        x: guipang(cfg=cfg['dataset_guipang'], part=x) for x in ['train', 'val']
    }
    # data_set = {
    #     x: dataset(cfg=cfg['dataset_qiafan'], part=x) for x in ['train', 'val']
    # }
    data_loader = {
        x: data.DataLoader(dataset[x], batch_size=cfg['batch_size'],
                           num_workers=4, shuffle=True, pin_memory=False)
        for x in ['train', 'val']
    }

    ######################################

    model=torchvision.models.detection.maskrcnn_resnet50_fpn()
    model.cuda()
    model=nn.DataParallel(model)

    criterion = {
        'crossentropyloss': nn.CrossEntropyLoss(),
        'hesloss': HESloss(),
        'mseloss': nn.MSELoss()
    }
    optimizer = optim.Adam(model.module.parameters(),
                           lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    
    best_model_wts = train_guipang(model, criterion, optimizer, scheduler, cfg)
    torch.save(best_model_wts, the_ckpt_root+'best.pkl')
