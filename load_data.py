from torch.utils.data import Dataset
import pandas as pd
import random
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import torch

IMG_PATH = "/media/cash/DATA/Dataset/CelebaFace/Img/img_celeba"


def preprocess_bbox_data(bbox):
    bbox = bbox.split(" ")
    bbox = [box.replace("\n", "") for box in bbox if box != '']
    
    data = {
        "img": bbox[0],
        "x": int(bbox[1]),
        "y": int(bbox[2]),
        "w": int(bbox[3]),
        "h": int(bbox[4])
        }
    
    return data


def preprocess_identity_data(identity):
    identity = identity.split(" ")
    identity = [i.replace("\n", "") for i in identity if i != '']
    
    data = {
        "img": identity[0],
        "label": int(identity[1]),
        }
    
    return data


def preprocessing():
    import os
    from PIL import Image
    
    imgs = os.listdir(IMG_PATH)
    file = imgs[0]
    Image.open(os.path.join(IMG_PATH, file))
    
    with open("/media/cashtsang/Toshiba/Dataset/CelebaFace/Anno/identity_CelebA.txt") as f:
        identities = f.readlines()
        
    with open("/media/cashtsang/Toshiba/Dataset/CelebaFace/Anno/list_bbox_celeba.txt") as f:
        bboxes = f.readlines()[2:]
    
    bboxes = list(map(preprocess_bbox_data, bboxes))
    identities = list(map(preprocess_identity_data, identities))
    
    import pandas as pd
    
    df_bbox = pd.DataFrame(bboxes)
    df_id = pd.DataFrame(identities)
    
    df = pd.merge(df_id, df_bbox, on='img')
    
    df.to_csv("./data/celeba.csv", index=False)
    


def train_test_split():
    
    df = pd.read_csv("./data/celeba.csv")
    df = df.sample(frac=1)
    df_len = len(df)
    split = int(0.9*df_len)
    
    train = df.iloc[:split]
    train.to_csv("./data/train_celeba.csv", index=False)
    
    valid = df.iloc[split:]
    valid.to_csv("./data/valid_celeba.csv", index=False)
    

    
class CelebaDetectionDataset(Dataset):
    
    def __init__(self, mode='train'):
        self.path = IMG_PATH
        
        if mode == "train":
            df = pd.read_csv("./data/train_celeba.csv")
        else:
            df = pd.read_csv("./data/valid_celeba.csv")
            
        self.data = df
        
        if mode == 'train':
            self.augmentation = transforms.Compose([
                transforms.RandomChoice([
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                    ], p=(0.1, 0.1, 0.1)),
                transforms.Resize((256, 256)),
                transforms.ToTensor()
                ])
        else:
            self.augmentation = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
                ])
    
        

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        img = Image.open(os.path.join(self.path, row['img']))
        width, height = img.size
        x, y, w, h = row['x'] / width, row['y'] / height, row['w'] / width, row['h'] / height
        
        img_tensor = self.augmentation(img)
        bbox = torch.tensor([[x, y, w, h]]).float()
        
        return img_tensor, bbox
        
        
    
    def __len__(self):
        return len(self.data)
    

class CelebaRecognitionDataset(Dataset):
    
    def __init__(self, mode='train'):
        self.path = IMG_PATH
        
        if mode == "train":
            df = pd.read_csv("./data/train_celeba.csv")
        else:
            df = pd.read_csv("./data/valid_celeba.csv")
            
        self.data = df
        self.identities = self.data['label'].unique()
        self.num_individual = len(self.identities)
        
        if mode == 'train':
            self.augmentation = transforms.Compose([
                transforms.RandomChoice([
                    # transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                    transforms.RandomHorizontalFlip(p=1.0)
                    ], p=(0.1, 0.1, 0.1)),
                transforms.Resize((256, 256)),
                transforms.ToTensor()
                ])
        else:
            self.augmentation = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
                ])
    

    def __getitem__(self, idx):
        label = self.identities[idx]
        
        temp = self.data.loc[self.data['label'] == label]     
        
        try:
            if len(temp) > 1:
                i, j = random.sample(list(range(0, len(temp))), k=2)
            else:
                i = j = 0
            
            row1 = temp.iloc[i]
            img1 = Image.open(os.path.join(self.path, row1['img']))
            x, y, w, h = int(row1['x']), int(row1['y']), int(row1['w']), int(row1['h'])
            crop1 = np.array(img1)[y:y+h,x:x+w,:]
            crop1 = self.augmentation(Image.fromarray(crop1))
            
            row2 = temp.iloc[j]
            img2 = Image.open(os.path.join(self.path, row2['img']))
            x, y, w, h = int(row2['x']), int(row2['y']), int(row2['w']), int(row2['h'])
            crop2 = np.array(img2)[y:y+h,x:x+w,:]
            crop2 = self.augmentation(Image.fromarray(crop2))
            return crop1, crop2
        
        except Exception as E:
            print(E)
            return self.__getitem__(idx)
        

        # return Image.fromarray(crop1), Image.fromarray(crop2)
        
        
    
    def __len__(self):
        return self.num_individual
    




if __name__ == "__main__":
    
    train_test_split()
    
