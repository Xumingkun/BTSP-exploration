import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
import numpy as np


class dataset_from_mat_dvs(data.Dataset):
    def __init__(self,file_path,wins):
        super(dataset_from_mat_dvs, self).__init__()
        self.wins=wins
        self.images=np.load(file_path+'/images.npy').astype(np.float32)[:,:,:,:,:wins]
        self.images=np.transpose(self.images,(0,4,3,1,2))
        self.images=self.images.reshape(-1,self.wins,34*34*2)
        self.label=np.load(file_path+'/labels.npy')
    def __getitem__(self, index):
        inputs=self.images[index,:,:]
        targets=self.label[index]
        inputs=torch.Tensor(inputs)
        return inputs,targets
    def __len__(self):
        return self.images.shape[0]