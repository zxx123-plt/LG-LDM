import numpy as np
from torch.utils import data
import glob
from torchvision import transforms
from PIL import Image
import os
from natsort import natsorted
from os.path import dirname as di
import torch
import sys;sys.path.append('./')
def get_image_address(dir):
    files = natsorted(os.listdir(dir))
    CFP_list = []
    for file_name in files:

        old_path = os.path.join(dir, file_name)

        base_name, picture_form = os.path.splitext(file_name)
        if os.path.isfile(old_path):
            if len(base_name.split('mask')) == 1:
                new_name = f'{base_name}{picture_form}'
                CFP_list.append(os.path.join(dir, new_name))
    return CFP_list

class FFATOFFA_dataset(data.Dataset):
    def __init__(self, data_path, img_size, 
                 mode='double', read_channel='color', 
                 data_aug=True):
        '''
        data_path: the up dir of data
        img_size: what size of image you want to read (tuple, int)
        mode: vary from: 1. 'double' 2. 'first' 3. 'second' 
        read_channel: 'color' or 'gray' 
        '''
        super(FFATOFFA_dataset, self).__init__()
        if isinstance(data_path, list):
            self.img_path = []
            for path in data_path:
                self.img_path += get_image_address(path)
        else:
            self.img_path = get_image_address(data_path)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        basic_trans_list = [
            transforms.ToTensor(),
            # transforms.Resize((512, 640), antialias=True),
            ]
        
        self.data_aug = data_aug
        if data_aug:
            self.augmentator = transforms.Compose([
                transforms.RandomRotation(5), 
                ##transforms.RandomCrop(img_size), 
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(), 
                transforms.Normalize(mean=0.5, std=0.5)
            ])
        else: 
            basic_trans_list.append(transforms.Resize(img_size, antialias=True))
            basic_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))
        self.transformer = transforms.Compose(basic_trans_list) 
        self.mode = mode
        if read_channel == 'color':
            self.img_reader = self.colorloader
        else:
            self.img_reader = self.grayloader

        self.CFP_reader = self.colorloader

        
    def double_get(self, CFP_path) -> list:
        parrent_dir = di(CFP_path)
        CFP_file = os.path.basename(CFP_path)
        num, suffix = os.path.splitext(CFP_file)
        FFA_path = os.path.join(parrent_dir, f'{num}.mask{suffix}')
        var_list = map(self.img_reader, [FFA_path])
        var_list = map(self.transformer, var_list)
        CFP = self.transformer(self.CFP_reader(CFP_path))
        if self.data_aug:
            var_list = torch.cat(list(var_list) + [CFP] )
            var_list = self.augmentator(var_list)
            FFA,CFP = var_list[0:1], var_list[1:4]
        else:
            FFA = var_list
            CFP = CFP
        return (CFP, FFA), CFP_file
    
    def __getitem__(self, index) -> list:
        CFP_name = self.img_path[index]
        paired_tuple, tuple_index = None, None
        if self.mode == 'double':
            paired_tuple, tuple_index = self.double_get(CFP_name)
        
        return paired_tuple, tuple_index
    
    def __len__(self):
        return len(self.img_path)
    
    def colorloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def grayloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
def FFAcondi_form_dataloader(dir, image_size, batch_size, mode, 
                           read_channel='color', data_aug=True, 
                           shuffle=True, drop_last=True, **kwargs):###**kwargs接受更多的参数
    dataset = FFATOFFA_dataset(dir, image_size, mode, read_channel, data_aug, **kwargs)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)



