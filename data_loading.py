import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as transforms
import PIL

class Load_from_path_Dataset(Dataset):
    def __init__(self,img_paths, home_dir, y, dim1=320, dim2=320):
        self.img_labels=y
        self.img_dir=home_dir
        self.img_paths=img_paths
        self.dim1=dim1
        self.dim2=dim2
        self.normalising=transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def __len__(self):
        return len(self.img_labels)
    
    def transformation(self, image, dim1, dim2):
        
        image=cv2.resize(image, (dim1, dim2))
        image = image / 255.
        image=torch.from_numpy(image)
        image = image.permute((2, 0, 1))
        image=self.normalising(image)
        
        return image

    
    def __getitem__(self, idx):
        img_path=os.path.join(self.img_dir+self.img_paths[idx])
        image=cv2.imread(img_path)
        image=self.transformation(image, self.dim1, self.dim2)

       
        # image=torch.from_numpy(image)
        # for i in range(2):
        #     image[i,:,:]=(image[i,:,:]-torch.mean(image[i,:,:]))/torch.std(image[i,:,:])
       
        label=self.img_labels[idx]
        label=torch.from_numpy(label)
        
        return image, label



