import copy
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from data import guipang
from data import qiafan
import pdb
# from utils import append_feature, calculate_map
import time
import shutil
from tensorboardX import SummaryWriter


def mapcal(epoch_map,outputs,targets,bar_scor,bar_iou):
# [{'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0'), 'masks': tensor([], device='cuda:0', size=(0, 1, 4096, 5500))}, 
# {'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0'), 'masks': tensor([], device='cuda:0', size=(0, 1, 4096, 5500
# {'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0'), 'masks': tensor([], device='cuda:0', size=(0, 1, 4096, 5500))}, {'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0'), 'masks': tensor([], device='cuda:0', size=(0, 1, 4096, 5500))}, 
# {'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0'), 'masks': tensor([], device='cuda:0', size=(0, 1, 4096, 5500))}]
    return 0.0


def train_guipang(model, criterion, optimizer, scheduler, cfg):

    best_map = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']):
        for phrase in ['train', 'val']:
            nowepochiter = (epoch-1)*int(data_set[phrase].__len__()/cfg['batch_size'])
            if phrase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            ##################################################
            # predic, gt = None, None
            epoch_map=0.0
            ##################################################

            for i in range(int(data_set[phrase].__len__()/cfg['batch_size'])):

                optimizer.zero_grad()
                images = []
                targets = []

                for ii in range(cfg['batch_size']):
                    data_set_return = data_set[phrase].__getitem__(
                        i*cfg['batch_size']+ii)
                    images.append(data_set_return[0].cuda())
                    targets.append({
                        'boxes': torch.cuda.FloatTensor([[int(data_set_return[1]['annotation']['object']['bndbox']['xmin']),
                                                          int(
                                                              data_set_return[1]['annotation']['object']['bndbox']['ymin']),
                                                          int(
                                                              data_set_return[1]['annotation']['object']['bndbox']['xmax']),
                                                          int(data_set_return[1]['annotation']['object']['bndbox']['ymax'])]]),
                        'labels': torch.cuda.LongTensor([1]),
                        'masks': torch.zeros(1, data_set_return[0].size()[1], data_set_return[0].size()[2])
                    })

                with torch.set_grad_enabled(phrase == 'train'):
                    outputs = model(images, targets)
                    if phrase == 'train':
                        loss = outputs['loss_classifier'] + \
                            outputs['loss_box_reg'] + \
                            outputs['loss_objectness'] + \
                            outputs['loss_rpn_box_reg']
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar('guipangtrain/loss',loss.item(), i+nowepochiter)                         
                    else:
                        epoch_map=mapcal(epoch_map,outputs,targets,bar_scor,bar_iou)
                        

            if phrase == 'val':
                if epoch_map > best_map:
                    best_map = epoch_map
                    best_model_wts = copy.deepcopy(model.module.state_dict())
                    torch.save(best_model_wts, the_ckpt_root+'best.pkl')
                if epoch % cfg['checkpoint'] == 0:
                    torch.save(copy.deepcopy(model.module.state_dict()),
                               the_ckpt_root+'{}.pkl'.format(epoch))
                
    return best_model_wts


if __name__ == '__main__':

    ######################################
    # config
    cfg = {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'batch_size': 5,
        'max_epoch': 100,
        'checkpoint': 20,
        'milestones': [30, 50],
        'gamma': 0.1,
        'dataset_guipang': '/repository/gong/qiafan/guipangdata/',
        'dataset_qiafan': '/repository/gong/qiafan/dataset/',
        'cuda_devices': '6',
        'ckpt_root': '/repository/gong/qiafan/'
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
    time_TrainStart = str(int(time.time()))
    the_ckpt_root = cfg['ckpt_root']+time_TrainStart+'/'
    os.mkdir(the_ckpt_root)
    # tensorboardx
    writer = SummaryWriter('runs/'+time_TrainStart+'_'+str(os.getpid()))

    ######################################
    data_set = {
        x: guipang(cfg=cfg['dataset_guipang'], part=x) for x in ['train', 'val']
    }
    # data_set = {
    #     x: dataset(cfg=cfg['dataset_qiafan'], part=x) for x in ['train', 'val']
    # }
    # data_loader = {
    #     x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'],
    #                        num_workers=4, shuffle=True, pin_memory=False)
    #     for x in ['train', 'val']
    # }
    ######################################

    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    model.cuda()
    model = nn.DataParallel(model)

    criterion = {
        'crossentropyloss': nn.CrossEntropyLoss(),
        'mseloss': nn.MSELoss()
    }
    optimizer = optim.Adam(model.module.parameters(),
                           lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])

    best_model_wts = train_guipang(model, criterion, optimizer, scheduler, cfg)
    torch.save(best_model_wts, the_ckpt_root+'best.pkl')






