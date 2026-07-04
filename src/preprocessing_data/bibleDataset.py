from torch.utils.data import Dataset
import torch.nn as nn
import os
from PIL import Image
from kraken.binarization import nlbin 

"""

The structure of the folder is : 
data/labelledData
    /images
        /test
        /train
        /val
    /labels
        /test
        /train
        /val
        
With each folder corresponding the to images and their respective labels

"""

class BibleDataset(Dataset) : 
    def __init__(self, root, split : str = "train", binarize : bool = False) : 
        self.root = root
        self.split = split
        self.binarize = binarize
        
        self.images_path = os.path.join(root, "images", split)
        self.labels_path = os.path.join(root, "labels", split)
        
        self.image_files = sorted(
            [file for file in os.listdir(self.images_path)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')) 
            and not file.startswith('.')
            ]
        )
    
    def load_image(self, path) : 
        image = Image.open(path).convert("RGB")
        
        if self.binarize:
            image = nlbin(image)
            
        return image
    
    def load_label(self, path) :  
        try : 
            with open(path, "r", encoding="utf-8") as f : 
                label = f.read().strip()
        except UnicodeDecodeError : 
            with open(path, "r", encoding="utf-16") as f : 
                label = f.read().strip()        
                
        return label
    
    def __getitem__(self, idx) : 
        image_filename = self.image_files[idx] 
        base_name = os.path.splitext(image_filename)[0]
        label_filename = f"{base_name}.gt.txt"
        
        image_path =  os.path.join(self.images_path, image_filename)
        label_path = os.path.join(self.labels_path, label_filename)
        
        image = self.load_image(image_path)
        label = self.load_label(label_path)
        
        return image, label
    
    def __len__(self) : 
        return len(self.image_files)