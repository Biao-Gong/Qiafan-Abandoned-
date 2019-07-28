import time
import os
import copy
import pdb
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import model
from data import guipang
from data import qiafan
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

assert torch.__version__.split('.')[1] == '4'

######################################
# config
cfg = {
	'lr': 0.00001,
	'momentum': 0.9,
	'weight_decay': 0.000005,
	'batch_size': 1,
	'max_epoch': 100,
	'checkpoint': 20,
	'milestones': [30, 50],
	'gamma': 0.1,
	'bar_scor': 0.7,
	'bar_iou': 0.7,
	'dataset_guipang': '/repository/gong/qiafan/guipangdata/',
	'dataset_qiafan': '/repository/gong/qiafan/dataset/',
	'cuda_devices': '4',
	'ckpt_root': '/repository/gong/qiafan/',
	'depth': 50
}

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
time_TrainStart = str(int(time.time()))
the_ckpt_root = cfg['ckpt_root']+time_TrainStart+'/'
os.mkdir(the_ckpt_root)
# tensorboardx
writer = SummaryWriter('runs/'+time_TrainStart+'_'+str(os.getpid()))


def main(args=None):

    data_set = {
        x: guipang(cfg=cfg['dataset_guipang'], part=x) for x in ['train', 'val']
    }
    # data_set = {
    #     x: qiafan(cfg=cfg['dataset_qiafan'], part=x) for x in ['train', 'val']
    # }
    data_loader = {
        x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'],
                           num_workers=4, shuffle=True, pin_memory=False)
        for x in ['train', 'val']
    }

	# Create the model
	if cfg['depth'] == 18:
		retinanet = model.resnet18(
		    num_classes=dataset_train.num_classes(), pretrained=True)
	elif cfg['depth'] == 34:
		retinanet = model.resnet34(
		    num_classes=dataset_train.num_classes(), pretrained=True)
	elif cfg['depth'] == 50:
		retinanet = model.resnet50(
		    num_classes=dataset_train.num_classes(), pretrained=True)
	elif cfg['depth'] == 101:
		retinanet = model.resnet101(
		    num_classes=dataset_train.num_classes(), pretrained=True)
	elif cfg['depth'] == 152:
		retinanet = model.resnet152(
		    num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError(
		    'Unsupported model depth, must be one of 18, 34, 50, 101, 152')

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet = torch.nn.DataParallel(retinanet).cuda()

	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	for epoch in range(1, cfg['max_epoch']):

		print('epoch: ', epoch)

		for phrase in ['train', 'val']:
            nowepochiter = (epoch-1)*len(data_loader[phrase])
			if phrase == 'train':
				scheduler.step()
				retinanet.training = True
				retinanet.train()
				retinanet.module.freeze_bn()
			else:
				retinanet.training = False
				retinanet.eval()

            ##################################################
            epoch_ap = 0.0
            ##################################################

			for i, (images, targets) in enumerate(data_loader[phrase]):
				optimizer.zero_grad()
				
				images = torch.cuda.FloatTensor(images.cuda())
				targets = torch.cuda.FloatTensor([list(map(float, data['annot']['annotation']['object']['bndbox']['xmin'])),
                        list(map(float, data['annot']['annotation']['object']['bndbox']['ymin'])),
                        list(map(float, data['annot']['annotation']['object']['bndbox']['xmax'])),
                        list(map(float, data['annot']['annotation']['object']['bndbox']['ymax']))])
				##################################################
				##################################################
				##################################################
                with torch.set_grad_enabled(phrase == 'train'):
                    if phrase == 'train':
						classification_loss, regression_loss = retinanet([images, targets])
						classification_loss = classification_loss.mean()
						regression_loss = regression_loss.mean()
						loss = classification_loss + regression_loss
						if bool(loss == 0):
							continue
						loss.backward()
						torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
						optimizer.step()
						writer.add_scalar('guipangtrain/loss',
                                      loss.item(), i+nowepochiter)
					else:
						pass
						# scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                        # epoch_ap = apcal_guipang(
                        #     epoch_ap, outputs, targets, cfg['bar_scor'], cfg['bar_iou'])

            if phrase == 'val':
                epoch_map = epoch_ap/float(data_set['val'].__len__())
                writer.add_scalar('guipangval/map', epoch_map, epoch)
                if epoch_map > best_map:
                    best_map = epoch_map
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, the_ckpt_root+'best.pkl')
                if epoch % cfg['checkpoint'] == 0:
                    torch.save(copy.deepcopy(model.state_dict()),
                               the_ckpt_root+'{}.pkl'.format(epoch))
					
if __name__ == '__main__':
	main()
