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

def test_guipang(model, cfg):

    phrase='val'
    model.eval()

    for i in range(int(data_set[phrase].__len__())):
        images = []
        data_set_return = data_set[phrase].__getitem__(i)
        images.append(data_set_return[0].cuda())
        outputs = model(images)

        ##################
        #draw




        writer.add_image()
        ###################



if __name__ == '__main__':

    ######################################
    # config
    cfg = {
        'batch_size': 1,
        'dataset_guipang': '/repository/gong/qiafan/guipangdata/',
        'dataset_qiafan': '/repository/gong/qiafan/dataset/',
        'cuda_devices': '6',
        'ckpt_root': '/repository/gong/qiafan/',
        'best_path':'111111'
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
    # tensorboardx
    writer = SummaryWriter('runs/'+'_predic_'+cfg['best_path'])


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
    model.module.load_state_dict(torch.load(os.path.join(cfg['ckpt_root'],cfg['best_path'])))
    test_guipang(model, cfg)






