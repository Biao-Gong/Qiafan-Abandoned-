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
from data import ModelNet40
# from models.MVHNet import *
# from utils import append_feature, calculate_map
import time
import shutil
from tensorboardX import SummaryWriter


def train_model_mesh(model, criterion, optimizer, scheduler, cfg):

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']):
        for phrase in ['train', 'test']:

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
                    lossa=criterion['crossentropyloss'](outputs, targets)
                    lossb=criterion[cfg['maskloss']](fea_hash_sensib_mesh, fea_hash_mesh)
                    # loss=lossa+lossa*lossb
                    loss=lossa+lossb
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

            if phrase == 'test':
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

    cfg = get_train_config(config_file='config/train_config.yaml')
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
    time_TrainStart = str(int(time.time()))
    the_ckpt_root = cfg['ckpt_root']+cfg['step']+time_TrainStart+'/'
    os.mkdir(the_ckpt_root)
    shutil.copyfile('./config/train_config.yaml',
                    the_ckpt_root+'train_config.yaml')

    # tensorboardx
    with open('training_log.json', 'a') as f:
        f.write('--------------------------\n')
        f.write('PID: '+str(os.getpid())+'\n')
        f.write('PWD: '+the_ckpt_root+'\n')
        f.write(str(json.dumps(cfg,indent=2))+'\n')
    writer = SummaryWriter('runs/'+cfg['step']+time_TrainStart+'_'+str(os.getpid()))

    ######################################

    # multi-modal dataset
    if cfg['modality'] == 'mesh':
        data_set = {
            x: ModelNet40(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
        }
    elif cfg['modality'] == 'view':
        data_set = {
            x: mv_ModelNet40(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
        }
    elif cfg['modality'] == 'meshview':
        data_set = {
            x: mesh_mv_ModelNet40(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
        }
    else:
        sys.exit('Wrong modality!')
    # data_loader
    data_loader = {
        x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'],
                           num_workers=4, shuffle=True, pin_memory=False)
        for x in ['train', 'test']
    }

    ######################################

    if cfg['step'] == 'meshhash':
        model = MeshHash(cfg=cfg)
        model.cuda()
        model = nn.DataParallel(model)

    elif cfg['step'] == 'mesh':
        model = MVHNet_mesh_TIP3(cfg=cfg)
        model.cuda()
        model = nn.DataParallel(model)
        model.module.multimodal_collection.load_state_dict(
            torch.load(cfg['meshnetparams_path']))
        model.module.hashmodal_collection.load_state_dict(
            torch.load(cfg['meshhashparams_path']))
        for param in model.module.multimodal_collection.parameters():
            param.requires_grad = False
        for param in model.module.hashmodal_collection.parameters():
            param.requires_grad = False

    elif cfg['step'] == 'viewhash':
        model = MVHash(cfg=cfg)
        model.cuda()
        model = nn.DataParallel(model)

    elif cfg['step'] == 'view':
        model = MVHNet_view_TIP(cfg=cfg)
        model.cuda()
        model = nn.DataParallel(model)
        model.module.multimodal_collection.load_state_dict(
            torch.load(cfg['viewparams_path']))
        model.module.hashmodal_collection.load_state_dict(
            torch.load(cfg['viewhashparams_path']))
        for param in model.module.multimodal_collection.parameters():
            param.requires_grad = False
        for param in model.module.hashmodal_collection.parameters():
            param.requires_grad = False

    else:
        sys.exit('Wrong step!')

    ######################################

    # fine-tune
    if cfg['loadmodelparams']:
        model.module.multimodal_collection.load_state_dict(
            torch.load(cfg['modelparams_path']))
        for param in model.module.multimodal_collection.parameters():
            param.requires_grad = True

    ######################################

    criterion = {
        'crossentropyloss': nn.CrossEntropyLoss(),
        'hesloss': HESloss(),
        'mseloss': nn.MSELoss()
    }
    criterion['hesloss'].cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters(
    )), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])

    ######################################

    if cfg['step'] == 'meshhash':
        best_model_wts = train_model_meshhash(model, criterion, optimizer, scheduler, cfg)
    elif cfg['step'] == 'mesh':
        best_model_wts = train_model_mesh(model, criterion, optimizer, scheduler, cfg)
    elif cfg['step'] == 'viewhash':
        best_model_wts = train_model_viewhash(model, criterion, optimizer, scheduler, cfg)
    elif cfg['step'] == 'view':
        best_model_wts = train_model_view(model, criterion, optimizer, scheduler, cfg)
    else:
        sys.exit('Wrong step!')

    ######################################
    
    torch.save(best_model_wts, the_ckpt_root+'best.pkl')
