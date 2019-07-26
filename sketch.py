# import os
# import torch.utils.data as data
# import copy
# import os
# import sys
# import json
import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision
# from config import get_train_config
# from models.MVHNet import *
# from utils import append_feature, calculate_map
# import time
# import shutil
# from tensorboardX import SummaryWriter
import pdb
dataset_guipang = '/repository/gong/qiafan/guipangdata/'
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


x=torch.tensor([1.0,2.0,0])
y=torch.tensor([[1.0,1.0,1.0,1.0],[2.0,2.0,5.0,5.0],[2.0,2.0,3.0,3.0]])
z=torch.tensor([[3.0,3.0,6.0,6.0]])



def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    #Intersection area
    # Intersection area 这里没有对inter_area为负的情况进行判断，后面计算出来的IOU就可能是负的
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1)*(b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1)*(b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


# def bb_intersection_over_union(boxA, boxB):
# 	# determine the (x, y)-coordinates of the intersection rectangle
# 	xA = max(boxA[0], boxB[0])
# 	yA = max(boxA[1], boxB[1])
# 	xB = min(boxA[2], boxB[2])
# 	yB = min(boxA[3], boxB[3])
 
# 	# compute the area of intersection rectangle
# 	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
# 	# compute the area of both the prediction and ground-truth
# 	# rectangles
# 	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
# 	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
# 	# compute the intersection over union by taking the intersection
# 	# area and dividing it by the sum of prediction + ground-truth
# 	# areas - the interesection area
# 	iou = interArea / float(boxAArea + boxBArea - interArea)
 
# 	# return the intersection over union value
# 	return iou


# def bb_intersection_over_union(boxA, boxB):
#     boxA = [int(x) for x in boxA]
#     boxB = [int(x) for x in boxB]

#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

#     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
#     iou = interArea / float(boxAArea + boxBArea - interArea)

#     return iou


# def IOU(Reframe,GTframe):
#     """
#     自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
#     """
#     x1 = Reframe[:,0]
#     y1 = Reframe[:,1]
#     width1 = Reframe[:,2]-Reframe[:,0]
#     height1 = Reframe[:,3]-Reframe[:,1]

#     x2 = GTframe[0]
#     y2 = GTframe[1]
#     width2 = GTframe[2]-GTframe[0]
#     height2 = GTframe[3]-GTframe[1]

#     endx = max(x1+width1,x2+width2)
#     startx = min(x1,x2)
#     width = width1+width2-(endx-startx)

#     endy = max(y1+height1,y2+height2)
#     starty = min(y1,y2)
#     height = height1+height2-(endy-starty)

#     if width <=0 or height <= 0:
#         ratio = 0 # 重叠率为 0 
#     else:
#         Area = width*height # 两矩形相交面积
#         Area1 = width1*height1
#         Area2 = width2*height2
#         ratio = Area*1./(Area1+Area2-Area)
#     # return IOU
#     return ratio,Reframe,GTframe

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

print(bbox_iou(y,z))



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
# data_set = {
#     x: guipang(cfg=cfg['dataset_guipang'], part=x) for x in ['train', 'val']
# }
# data_set = {
#     x: dataset(cfg=cfg['dataset_qiafan'], part=x) for x in ['train', 'val']
# }
# data_loader = {
#     x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'],
#                         num_workers=4, shuffle=True, pin_memory=False)
#     for x in ['train', 'val']
# }
# a=[1,2]
# print(a)
# pdb.set_trace()

# os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']


# model = torchvision.models.detection.maskrcnn_resnet50_fpn()
# model.cuda()
# model = nn.DataParallel(model)
# model.train()


# phrase = 'train'
# for i in range(int(data_set[phrase].__len__()/cfg['batch_size'])):
#     images = []
#     targets = []
#     for ii in range(cfg['batch_size']):
#         data_set_return = data_set[phrase].__getitem__(i*cfg['batch_size']+ii)
#         images.append(data_set_return[0].cuda())
#         targets.append({
#             'boxes': torch.cuda.FloatTensor([[int(data_set_return[1]['annotation']['object']['bndbox']['xmin']),
#                                               int(data_set_return[1]['annotation']['object']['bndbox']['ymin']),
#                                               int(data_set_return[1]['annotation']['object']['bndbox']['xmax']),
#                                               int(data_set_return[1]['annotation']['object']['bndbox']['ymax'])]]),
#             'labels': torch.cuda.LongTensor([1])
#         })
#         # print(ii)
#         print(data_set_return[0].size()[1])
#         pdb.set_trace()
#     print(images)
#     print(targets)




    # print(data_set[phrase].__getitem__(i))
    # print(cfg['batch_size'])
    # print(i)
    # print(images)
    # print(annotations)

    # print(annotations['annotation']['object']['name'])
    # annotations=torch.FloatTensor(list(map(int,annotations['annotation']['object']['name'])))
    # print(annotations['annotation']['object']['bndbox']['xmin'])
    # print(annotations['annotation']['object']['bndbox'])

    # print(torch.tensor([list(map(int,annotations['annotation']['object']['bndbox']['xmin'])),
    #     list(map(int,annotations['annotation']['object']['bndbox']['ymin']))]).t())
    # pdb.set_trace()

    # images = torch.cuda.FloatTensor(images.cuda())
    # targets = []

    # boxes = torch.cuda.FloatTensor([list(map(int,annotations['annotation']['object']['bndbox']['xmin'])),
    #                                 list(map(int,annotations['annotation']['object']['bndbox']['ymin'])),
    #                                 list(map(int,annotations['annotation']['object']['bndbox']['xmax'])),
    #                                 list(map(int,annotations['annotation']['object']['bndbox']['ymax']))]).t()
    # labels = torch.cuda.LongTensor(
    #     list(map(int, annotations['annotation']['object']['name'])))

    # with torch.set_grad_enabled(phrase == 'train'):
    #     outputs = model(images,boxes=boxes,labels=labels)
    #     print(outputs)
    #     pdb.set_trace()
