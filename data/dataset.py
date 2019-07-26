import numpy as np
import os
import torch
import torch.utils.data as data
import sys
import collections

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


import glob
import random
from PIL import Image
from torchvision import transforms
from itertools import groupby



def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

class guipang(data.Dataset):
    def __init__(self, cfg, part='train'):
        self.part = part
        self.root = cfg
        self.images=[]
        self.annotations=[]
        self.transforms = transforms.Compose([
            # transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])
        for filename in os.listdir(os.path.join(self.root,self.part)):
            if os.path.splitext(filename)[1]=='.jpg':
                self.images.append(os.path.join(self.root,self.part,filename))
                self.annotations.append(os.path.join(self.root,self.part,os.path.splitext(filename)[0]+'.xml'))
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

























class guipang1(data.Dataset):
    def __init__(self, cfg, part='train'):
        self.part = part
        self.root = cfg
        self.images=[]
        self.annotations=[]
        self.transforms = transforms.Compose([
            # transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])
        # self.data = []
        # type_index = 0
        # for type in os.listdir(self.pcroot):
        #     type_index = self.class_order.index(type)
        #     type_root = os.path.join(os.path.join(self.pcroot, type), set)
        #     for filename in os.listdir(type_root):
        #         if filename.endswith('.npy'):
        #             self.data.append(
        #                 (os.path.join(type_root, filename), type_index))
        #     type_index += 1
        for filename in os.listdir(os.path.join(self.root,self.part)):
            if os.path.splitext(filename)[1]=='.jpg':
                self.images.append(os.path.join(self.root,self.part,filename))
                self.annotations.append(os.path.join(self.root,self.part,os.path.splitext(filename)[0]+'.xml'))
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


class qiafan(data.Dataset):
    def __init__(self, cfg, set='train'):
        self.set = set
        self.pcroot = cfg['data_root_pc']
        self.data = []
        self.class_order = cfg['class_order']
        type_index = 0
        for type in os.listdir(self.pcroot):
            type_index = self.class_order.index(type)
            type_root = os.path.join(os.path.join(self.pcroot, type), set)
            for filename in os.listdir(type_root):
                if filename.endswith('.npy'):
                    self.data.append(
                        (os.path.join(type_root, filename), type_index))
            type_index += 1

    def __getitem__(self, i):
        path, type = self.data[i]
        raw_pos = np.load(path)
        pos = torch.from_numpy(raw_pos[:1024]).float()
        fea = torch.ones(1024, 1)
        target = torch.tensor(type, dtype=torch.long)
        return (pos, fea), target

    def __len__(self):
        return len(self.data)

















class pc_ModelNet40(data.Dataset):
    def __init__(self, cfg, set='train'):
        self.set = set
        self.pcroot = cfg['data_root_pc']
        self.data = []
        self.class_order = cfg['class_order']
        type_index = 0
        for type in os.listdir(self.pcroot):
            type_index = self.class_order.index(type)
            type_root = os.path.join(os.path.join(self.pcroot, type), set)
            for filename in os.listdir(type_root):
                if filename.endswith('.npy'):
                    self.data.append(
                        (os.path.join(type_root, filename), type_index))
            type_index += 1

    def __getitem__(self, i):
        path, type = self.data[i]
        raw_pos = np.load(path)
        pos = torch.from_numpy(raw_pos[:1024]).float()
        fea = torch.ones(1024, 1)
        target = torch.tensor(type, dtype=torch.long)
        return (pos, fea), target

    def __len__(self):
        return len(self.data)


# class mv_ModelNet40(data.Dataset):
#     def __init__(self, cfg, status="train"):
#         super(ModelNet40, self).__init__()
#         self.data_root = cfg['data_root_mv']
#         self.status = status
#         self.img_size = cfg['img_size']
#         self.views_list = []
#         self.label_list = []
#         for i, curr_category in enumerate(sorted(get_immediate_subdirectories(self.data_root))):
#             if status == "test":
#                 working_dir = os.path.join(
#                     self.data_root, curr_category, 'test')
#             elif status == "train":
#                 working_dir = os.path.join(
#                     self.data_root, curr_category, 'train')
#             else:
#                 raise NotImplementedError
#             all_img_list = glob.glob(working_dir + "/*.jpg")
#             append_views_list = [[v for v in g] for _, g in groupby(
#                 sorted(all_img_list), lambda x: x.split('_')[-2])]
#             self.views_list += append_views_list
#             self.label_list += [i] * len(append_views_list)
#         assert len(self.views_list) == len(self.label_list)
#         self.transform = transforms.Compose([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor()
#         ])

#     def __getitem__(self, index):
#         views = [self.transform(Image.open(v)) for v in self.views_list[index]]
#         return torch.stack(views), self.label_list[index]

#     def __len__(self):
#         return len(self.views_list)


class mesh_ModelNet40(data.Dataset):

    def __init__(self, cfg, part='train'):
        self.root = cfg['data_root']
        self.augment_data = cfg['augment_data']
        self.part = part

        self.data = []
        type_index = 0
        for type in os.listdir(self.root):
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append(
                        (os.path.join(type_root, filename), type_index))
            type_index += 1

    def __getitem__(self, i):
        path, type = self.data[i]
        data = np.load(path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # data augmentation
        if self.augment_data and self.part == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data = np.clip(
                sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face = np.concatenate(
                (face[:, :12] + jittered_data, face[:, 12:]), 1)

        # fill for n < 1024
        num_point = len(face)
        if num_point < 1024:
            fill_face = []
            fill_neighbor_index = []
            for i in range(1024 - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate(
                (neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        target = torch.tensor(type, dtype=torch.long)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)


class ModelNet40(data.Dataset):

    def __init__(self, cfg, part='train'):
        self.root = cfg['data_root']
        self.augment_data = cfg['augment_data']
        self.part = part
        self.class_order=cfg['class_order']


        self.data = []
        type_index = 0
        for type in os.listdir(self.root):
            #print('type:',type)
            type_index=self.class_order.index(type)
            #print(type_index)
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append((os.path.join(type_root, filename), type_index))
            #type_index += 1
        #print('datalong',len(self.data))

    def __getitem__(self, i):
        path, type = self.data[i]
        #print('path:',path)
        data = np.load(path)
        if 'face' in data.keys():
            face = data['face']
        else:
            face = data['faces']
        if 'neighbors' in data.keys():
            neighbor_index = data['neighbors']
        else:
            neighbor_index = data['neighbor_index']

        # data augmentation
        if self.augment_data and self.part == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

        # fill for n < 1024
        num_point = len(face)
        if num_point < 4096:
            fill_face = []
            fill_neighbor_index = []
            for i in range(4096 - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))


        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        target = torch.tensor(type, dtype=torch.long)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)


##############################################################


class mesh_mv_ModelNet40(data.Dataset):

    def __init__(self, cfg, part='train'):
        self.root = cfg['data_root']
        self.mvroot = cfg['data_root_mv']
        self.augment_data = cfg['augment_data']
        self.img_size = cfg['img_size']
        self.part = part
        self.class_order=cfg['class_order']
        
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        self.data = []
        type_index = 0
        for type in os.listdir(self.root):
            type_index = self.class_order.index(type)
            # print('type:',type_index)   ####读取类别
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append(
                        (os.path.join(type_root, filename), type_index))
            # type_index += 1

    def __getitem__(self, i):
        path, type = self.data[i]
        data = np.load(path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # data augmentation
        if self.augment_data and self.part == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data = np.clip(
                sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face = np.concatenate(
                (face[:, :12] + jittered_data, face[:, 12:]), 1)


        # fill for n < 1024
        # num_point = len(face)
        # if num_point < 1024:
        #     fill_face = []
        #     fill_neighbor_index = []
        #     for i in range(1024 - num_point):
        #         index = np.random.randint(0, num_point)
        #         fill_face.append(face[index])
        #         fill_neighbor_index.append(neighbor_index[index])
        #     face = np.concatenate((face, np.array(fill_face)))
        #     neighbor_index = np.concatenate(
        #         (neighbor_index, np.array(fill_neighbor_index)))

        num_point = len(face)
        if num_point < 4096:
            fill_face = []
            fill_neighbor_index = []
            for i in range(4096 - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        target = torch.tensor(type, dtype=torch.long)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        ##############################################################
        # read views
        mv_path = str(path).replace(
            'mesh_ModelNet40', self.mvroot.split('/')[-2])
        mv_path = mv_path.replace('.npz', '_*.jpg')
        img_list = glob.glob(mv_path)
        views = [self.transform(Image.open(v)) for v in img_list]
        for i in range(len(views)):
            views[i] = views[i].reshape(1, views[i].size(
                0), views[i].size(1), views[i].size(2))
            if i == 0:
                return_views = views[i]
            else:
                return_views = torch.cat((return_views, views[i]))
        views = return_views

        return views, centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)


class mv_ModelNet40(data.Dataset):

    def __init__(self, cfg, part='train'):
        self.root = cfg['data_root']
        self.mvroot = cfg['data_root_mv']
        self.augment_data = cfg['augment_data']
        self.img_size = cfg['img_size']
        self.part = part
        self.class_order=cfg['class_order']
        
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        self.data = []
        type_index = 0
        for type in os.listdir(self.root):
            type_index = self.class_order.index(type)
            # print('type:',type_index)   ####读取类别
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append(
                        (os.path.join(type_root, filename), type_index))
            # type_index += 1

    def __getitem__(self, i):
        path, type = self.data[i]
        data = np.load(path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # # data augmentation
        # if self.augment_data and self.part == 'train':
        #     sigma, clip = 0.01, 0.05
        #     jittered_data = np.clip(
        #         sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
        #     face = np.concatenate(
        #         (face[:, :12] + jittered_data, face[:, 12:]), 1)


        # # fill for n < 1024
        # # num_point = len(face)
        # # if num_point < 1024:
        # #     fill_face = []
        # #     fill_neighbor_index = []
        # #     for i in range(1024 - num_point):
        # #         index = np.random.randint(0, num_point)
        # #         fill_face.append(face[index])
        # #         fill_neighbor_index.append(neighbor_index[index])
        # #     face = np.concatenate((face, np.array(fill_face)))
        # #     neighbor_index = np.concatenate(
        # #         (neighbor_index, np.array(fill_neighbor_index)))

        # num_point = len(face)
        # if num_point < 4096:
        #     fill_face = []
        #     fill_neighbor_index = []
        #     for i in range(4096 - num_point):
        #         index = np.random.randint(0, num_point)
        #         fill_face.append(face[index])
        #         fill_neighbor_index.append(neighbor_index[index])
        #     face = np.concatenate((face, np.array(fill_face)))
        #     neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # # to tensor
        # face = torch.from_numpy(face).float()
        # neighbor_index = torch.from_numpy(neighbor_index).long()
        # target = torch.tensor(type, dtype=torch.long)

        # # reorganize
        # face = face.permute(1, 0).contiguous()
        # centers, corners, normals = face[:3], face[3:12], face[12:]
        # corners = corners - torch.cat([centers, centers, centers], 0)

        ##############################################################
        # read views
        mv_path = str(path).replace(
            'mesh_ModelNet40', self.mvroot.split('/')[-2])
        mv_path = mv_path.replace('.npz', '_*.jpg')
        img_list = glob.glob(mv_path)
        views = [self.transform(Image.open(v)) for v in img_list]
        for i in range(len(views)):
            views[i] = views[i].reshape(1, views[i].size(
                0), views[i].size(1), views[i].size(2))
            if i == 0:
                return_views = views[i]
            else:
                return_views = torch.cat((return_views, views[i]))
        views = return_views

        return views, target

    def __len__(self):
        return len(self.data)
