import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
from natsort import natsorted
import torch
import sys;sys.path.append('./')

def get_multi_image_addresses(dir):

    files = natsorted(os.listdir(dir))
    

    file_groups = {}
    
    for file_name in files:
        old_path = os.path.join(dir, file_name)
        if os.path.isfile(old_path):
            base_name, ext = os.path.splitext(file_name)
            

            if '.' in base_name:
                key = base_name.split('.')[0]  
                suffix = base_name.split('.')[1]  
                
                if key not in file_groups:
                    file_groups[key] = {}
                
                file_groups[key][suffix] = old_path
    

    complete_groups = []
    required_types = {'ffa', 'cfp', 'od', 'vessel'}
    
    for key, group in file_groups.items():
        if required_types.issubset(set(group.keys())):
            complete_groups.append({
                'ffa': group['ffa'],
                'cfp': group['cfp'], 
                'od': group['od'],
                'vessel': group['vessel']
            })
    
    return complete_groups

class ConditionalResize(object):
   
    def __init__(self, target_size=(512, 512), interpolation=Image.BICUBIC):
        self.target_size = target_size
        self.interpolation = interpolation
    
    def __call__(self, img):
        current_size = img.size  # PIL Image.size返回 (width, height)
        
        if current_size == self.target_size:
            return img
        else:
            return img.resize(self.target_size, self.interpolation)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(target_size={self.target_size}, interpolation={self.interpolation})"

class MultiImage_dataset(data.Dataset):
    def __init__(self, data_path, img_size, 
                 data_aug=True, force_resize_to_512=True, mode='train'):

        super(MultiImage_dataset, self).__init__()
        if isinstance(data_path, list):
            self.img_groups = []
            for path in data_path:
                self.img_groups += get_multi_image_addresses(path)
        else:
            self.img_groups = get_multi_image_addresses(data_path)
            
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        self.mode = mode
        

        basic_trans_list = []
        

        if force_resize_to_512:
            basic_trans_list.append(ConditionalResize(target_size=(512, 512), 
                                                    interpolation=Image.BICUBIC))

        basic_trans_list.append(transforms.ToTensor())
        
        self.data_aug = data_aug and mode == 'train'
        if self.data_aug:

            if not force_resize_to_512:
                basic_trans_list.append(transforms.Resize(img_size, 
                                                        interpolation=transforms.InterpolationMode.BICUBIC, 
                                                        antialias=True))
            

            self.ffa_cfp_augmentator = transforms.Compose([
                transforms.RandomRotation(5), 
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(), 
                transforms.Normalize(mean=0.5, std=0.5)
            ])
            

            self.binary_augmentator = transforms.Compose([
                transforms.RandomRotation(5), 
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip()
            ])
        else: 

            if not force_resize_to_512:
                basic_trans_list.append(transforms.Resize(img_size, 
                                                        interpolation=transforms.InterpolationMode.BICUBIC,
                                                        antialias=True))
            

            ffa_cfp_trans_list = basic_trans_list.copy()
            ffa_cfp_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))
            self.ffa_cfp_transformer = transforms.Compose(ffa_cfp_trans_list)
            

            self.binary_transformer = transforms.Compose(basic_trans_list)
        

        if self.data_aug:
            self.transformer = transforms.Compose(basic_trans_list)
        


    def __getitem__(self, index):
        img_group = self.img_groups[index]
        
        # 获取FFA文件名（用作标识）
        ffa_name = os.path.basename(img_group['ffa'])
        
        # 读取四种类型的图像
        ffa_image = self.grayloader(img_group['ffa'])      
        cfp_image = self.colorloader(img_group['cfp'])     
        od_image = self.grayloader(img_group['od'])        
        vessel_image = self.grayloader(img_group['vessel']) 
        
        if self.data_aug:

            ffa_tensor = self.transformer(ffa_image)
            cfp_tensor = self.transformer(cfp_image)
            od_tensor = self.transformer(od_image)
            vessel_tensor = self.transformer(vessel_image)
            

            seed = torch.randint(0, 2**32, (1,)).item()
            

            torch.manual_seed(seed)
            ffa_tensor = self.ffa_cfp_augmentator(ffa_tensor)
            torch.manual_seed(seed)
            cfp_tensor = self.ffa_cfp_augmentator(cfp_tensor)
            

            torch.manual_seed(seed)
            od_tensor = self.binary_augmentator(od_tensor)
            torch.manual_seed(seed)
            vessel_tensor = self.binary_augmentator(vessel_tensor)
            
        else:

            ffa_tensor = self.ffa_cfp_transformer(ffa_image)
            cfp_tensor = self.ffa_cfp_transformer(cfp_image)
            od_tensor = self.binary_transformer(od_image)
            vessel_tensor = self.binary_transformer(vessel_image)
        
        return ffa_tensor, cfp_tensor, od_tensor, vessel_tensor, ffa_name

    def __len__(self):
        return len(self.img_groups)

    def colorloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def grayloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

def FFAcondi_form_dataloader(dir, image_size, batch_size, mode='train',
                            data_aug=True, shuffle=True, drop_last=True, 
                            force_resize_to_512=True, read_channel='gray', **kwargs):

    dataset = MultiImage_dataset(dir, image_size, data_aug, 
                                force_resize_to_512, mode, **kwargs)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)