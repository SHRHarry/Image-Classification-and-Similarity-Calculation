# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:33:21 2023

@author: ms024
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    
    def __init__(self, root_dir, mode = "train", transforms = None):
        
        super().__init__()
        self.root_dir = os.path.join(root_dir, mode)
        self.imgs = []
        self.labels = []
        self.class_list = os.listdir(self.root_dir)
        for class_idx in self.class_list:
            class_folder = os.path.join(self.root_dir, class_idx)
            img_list = os.listdir(class_folder)
            for img_name in img_list:
                self.imgs.append(img_name)
                self.labels.append(class_idx)
        
        self.mode = mode
        self.transforms = transforms
        
    def __getitem__(self, idx):
        
        image_name = self.imgs[idx]
        label = self.labels[idx]
        
        
        img = cv2.imread(os.path.join(self.root_dir, label, image_name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        if self.mode == "train" or self.mode == "val":
        
            ### Preparing class label
            label = self.class_list.index(label)
            # label = torch.tensor(label, dtype = torch.float32)
            label = torch.tensor([label], dtype = torch.float32)

            ### Apply Transforms on image
            img = self.transforms(img)

            return img, label
        
        elif self.mode == "test":
            
            ### Apply Transforms on image
            img = self.transforms(img)

            return img
            
        
    def __len__(self):
        return len(self.imgs)