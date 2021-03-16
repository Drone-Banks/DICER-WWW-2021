import numpy as np
import torch

class PointDataset(torch.utils.data.Dataset):
    """
    Rating Dataset
    return format: 
        Label, UserID, TargetID
    """

    def __init__(self, dataset):
        dataset = np.array(dataset)
        self.users = dataset[:, 0]
        self.candidates = dataset[:, 1]
        self.labels = dataset[:, 2]

    def __getitem__(self, index):
        return [self.labels[index], self.users[index], self.candidates[index]]

    def __len__(self):
        return len(self.labels)

class RankDataset(torch.utils.data.Dataset):
    """
    Rating Dataset
    format: 
        UserID, TargetID
    """

    def __init__(self, dataset):
        dataset = np.array(dataset)
        self.users = dataset[:, 0]
        self.candidates = dataset[:, 1]

    def __getitem__(self, index):
        return [self.users[index], self.candidates[index]]

    def __len__(self):
        return len(self.users)



