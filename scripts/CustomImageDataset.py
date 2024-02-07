from skimage.color import rgb2lab
from skimage.io import imread
from torch.utils.data import Dataset
from torch import reshape

import os
from torch import from_numpy
from torch import permute


class CustomImageDataset(Dataset):
    def __init__(self, img_dir:str, n_images=None):
        self.img_dir = img_dir
        self._img_files = os.listdir(self.img_dir)
        self._img_files = self._img_files if n_images is None else self._img_files[0:n_images]
    
    def __len__(self) -> int:
        return len(self._img_files)
    
    def __getitem__(self, idx):
        img_name = self._img_files[idx]
        img = imread(f"{self.img_dir}/{img_name}")
        try:
            img = rgb2lab(img)
        except:
            print(f"{self.img_dir}/{img_name}")
            print(img.shape)
            raise ValueError
        img = (img + [0, 128, 128]) / [100, 255, 255]
        img = from_numpy(img).float()
        img = permute(img, (2,0,1))
        return reshape(img[0,:,:], (1,256,256)), img[1:,:,:]
        
