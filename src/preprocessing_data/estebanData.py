from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image

class EstebanData(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):

        self.root = root
        self.split = split
        self.transform = transform
        
        self.image_path = os.path.join(root, "all_bin")
        self.label_file = os.path.join(root, "gray_labels.xlsx")
        
        df = pd.read_excel(self.label_file)
        
        mask = (df['split'] == self.split)
        filtered_df = df.loc[mask]
        
        self.filenames = filtered_df['preproc_file_name'].values
        self.labels = filtered_df['label'].values
        
        # Create the lookup mapping
        self.label_map = dict(zip(self.filenames, self.labels))

    def get_label(self, image_filename: str) -> str:
        try : 
            label = self.label_map.get(image_filename)
            return label
        except KeyError:
            return f"Label for {image_filename} not found."

    def get_image(self, image_filename: str):
        img_path = os.path.join(self.image_path, image_filename)
        
        try : 
            image = Image.open(img_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        fname = self.filenames[index]

        image = self.get_image(fname)
        label = self.get_label(fname)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
