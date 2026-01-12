import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
from natsort import natsorted
import torch
import sys;sys.path.append('./')

def get_ffa_image_address(dir):

    files = natsorted(os.listdir(dir))
    FFA_list = []

    for file_name in files:
        old_path = os.path.join(dir, file_name)


        if not os.path.isfile(old_path):
            continue

        if 'ffa' in file_name.lower():
            FFA_list.append(old_path)

    return FFA_list

class ConditionalResize(object):

    def __init__(self, target_size=(512, 512), interpolation=Image.BICUBIC):

        self.target_size = target_size
        self.interpolation = interpolation
    
    def __call__(self, img):

        # 获取当前图像大小
        current_size = img.size  # PIL Image.size返回 (width, height)
        
        # 如果已经是目标大小，直接返回
        if current_size == self.target_size:
            return img
        else:
            return img.resize(self.target_size, self.interpolation)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(target_size={self.target_size}, interpolation={self.interpolation})"

class FFA_dataset(data.Dataset):
    def __init__(self, data_path, img_size, 
                 read_channel='gray', data_aug=True, force_resize_to_512=True):

        super(FFA_dataset, self).__init__()
        if isinstance(data_path, list):
            self.img_path = []
            for path in data_path:
                self.img_path += get_ffa_image_address(path)
        else:
            self.img_path = get_ffa_image_address(data_path)
            
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        

        basic_trans_list = []
        

        if force_resize_to_512:
            basic_trans_list.append(ConditionalResize(target_size=(512, 512), 
                                                    interpolation=Image.BICUBIC))
        

        basic_trans_list.append(transforms.ToTensor())
        
        self.data_aug = data_aug
        if data_aug:

            if not force_resize_to_512:
                basic_trans_list.append(transforms.Resize(img_size, 
                                                        interpolation=transforms.InterpolationMode.BICUBIC, 
                                                        antialias=True))
            
            self.augmentator = transforms.Compose([
                transforms.RandomRotation(5), 
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(), 
                transforms.Normalize(mean=0.5, std=0.5)
            ])
        else: 

            if not force_resize_to_512:
                basic_trans_list.append(transforms.Resize(img_size, 
                                                        interpolation=transforms.InterpolationMode.BICUBIC,
                                                        antialias=True))
            basic_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))
        
        self.transformer = transforms.Compose(basic_trans_list)
        
        if read_channel == 'color':
            self.img_reader = self.colorloader
        else:
            self.img_reader = self.grayloader
        
        # 添加属性以便调试
        self.force_resize_to_512 = force_resize_to_512


    def __getitem__(self, index):
        ffa_path = self.img_path[index]
        ffa_name = os.path.basename(ffa_path)
        

        ffa_image = self.img_reader(ffa_path)

        if hasattr(self, '_debug_printed') and not self._debug_printed:
            print(f"样本图像大小 - FFA: {ffa_image.size}")
            self._debug_printed = True

        ffa_tensor = self.transformer(ffa_image)

        if self.data_aug:
            ffa_tensor = self.augmentator(ffa_tensor)
        
        return ffa_tensor, ffa_name

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

def ffa_dataloader(dir, image_size, batch_size,
                   read_channel='gray', data_aug=True,
                   shuffle=True, drop_last=True, 
                   force_resize_to_512=True, **kwargs):

    dataset = FFA_dataset(dir, image_size, read_channel, data_aug, 
                         force_resize_to_512=force_resize_to_512, **kwargs)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)


