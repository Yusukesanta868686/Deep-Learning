import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as trasforms
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, directory, mode, transform = None, csv_path = None):
        self.directory = directory
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []

        if mode == "train":
            for label_dir in os.listdir(directory):
                class_path = os.path.join(directory, label_dir)
                
                if os.path.isdir(class_path):
                    for image_file in os.listdir(class_path):
                        if image_file.endswith(".jpg") or image_file.endswith(".png"):
                            self.images.append(os.path.join(label_dir, image_file))
                            self.labels.append(int(label_dir[1]))
            
        elif mode == "test":
            directory_img = os.path.join(directory, 'imgs/test')
            
            csv_path = os.path.join(directory, csv_path)
            test_data = pd.read_csv(csv_path)
            #print(test_data["img"].str.extract('(\d+)').head())
            test_data['class_num'] = test_data['classname'].str.extract('(\d+)').astype(int)
            test_data['img_num'] = test_data['img'].str.extract('(\d+)').astype(int)
        
            for _, row in test_data.iterrows():
                image_file = row['img']
                image_path = os.path.join(directory_img, image_file) 
                
                if image_file.endswith(".jpg") or image_file.endswith(".png"):
                    self.images.append(os.path.join("imgs/test", image_file))
                    self.labels.append(row['class_num'])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
    
        return image, label
    