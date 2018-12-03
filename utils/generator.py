import numpy as np
from torch.utils.data import sampler, TensorDataset, Dataset
import random
from utils.img_preprocess import img_process_PIL

class kidney_Dataset(Dataset):
    def __init__(self, train_id, df, train, seq, transformation, transform = True, size = (224, 224)):
        self.train_id = train_id
        self.df = df
        self.train = train
        self.seq = seq
        self.transformation = transformation
        self.transform = transform
        self.size = size
    def __getitem__(self, index):
        sample_count = len(self.df[self.df['uid_date'] == self.train_id[index]]['path'].values) # img for all img
        length = self.df[self.df['uid_date'] == self.train_id[index]]['length'].values
        
        if self.transform:
            select_index = random.randint(0, sample_count-1)
        else:
            select_index = np.where(length == np.max(length))[0][0]
                
        path = self.df[self.df['uid_date'] == self.train_id[index]]['path'].values[select_index]
        img = img_process_PIL(path, self.seq, self.transformation, self.transform, self.size)
        label = self.df[self.df['uid_date'] == self.train_id[index]]['egfr_mdrd'].values[select_index]
        return img, label

    def __len__(self):
        return len(self.train_id)