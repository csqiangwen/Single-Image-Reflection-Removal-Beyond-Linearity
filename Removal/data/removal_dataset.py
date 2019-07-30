import os.path
import torch.utils.data as data
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np
import torch


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

class RemovalDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        if opt.phase == 'train':
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
            self.dir_W = os.path.join(opt.dataroot, opt.phase + 'W')

        self.C_paths = make_dataset(self.dir_C)
        if opt.phase == 'train':
            self.A_paths = make_dataset(self.dir_A)
            self.B_paths = make_dataset(self.dir_B)
            self.W_paths = make_dataset(self.dir_W)

        self.C_paths = sorted(self.C_paths)
        if opt.phase == 'train':
            self.A_paths = sorted(self.A_paths)
            self.B_paths = sorted(self.B_paths)
            self.W_paths = sorted(self.W_paths)

        self.C_size = len(self.C_paths)

    def get_transforms_0(self, img, i, j):
        img = transforms.functional.crop(img, i, j, 256, 256)

        return img

    def get_transforms_1(self, img):
        transform = transforms.CenterCrop(512)
        img = transform(img)

        return img

    def get_transforms_2(self, img):
        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        img = transform(img)

        return img

    def __getitem__(self, index):
        C_path = self.C_paths[index]
        if self.opt.phase == 'train':
            A_path = self.A_paths[index%self.C_size]
            B_path = self.B_paths[index%self.C_size]
            W_path = self.W_paths[index%self.C_size]

        C_img = Image.open(C_path).convert('RGB')
        if self.opt.phase == 'train':
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            W_np = np.load(W_path)
        
        C = self.get_transforms_2(C_img)
        if self.opt.phase == 'train':
            A = self.get_transforms_2(A_img)
            B = self.get_transforms_2(B_img)
            W = torch.from_numpy(W_np).view(3, self.opt.loadSize, self.opt.loadSize)

        if self.opt.phase == 'train':
            return {'A': A, 'B': B, 'C': C, 'W': W,
                    'C_path': C_path}
        return {'C': C, 'C_path': C_path}

    def __len__(self):
        return self.C_size

    def name(self):
        return 'RemovalDataset'
