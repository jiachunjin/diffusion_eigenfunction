import os
import torch


class toy_sin_pair():
    def __init__(self, root='./data'):
        data_path = os.path.join(root, 'toy_sin', 'data.pt')
        data = torch.load(data_path)
        self.data_a = data['x_a']
        self.data_b = data['x_b']
        assert len(self.data_a) == len(self.data_b)
    
    def __getitem__(self, index):
        return (self.data_a[index], self.data_b[index]), -1

    def __len__(self):
        return len(self.data_a)

class toy_sin_a():
    def __init__(self, root='./data'):
        data_path = os.path.join(root, 'toy_sin', 'data.pt')
        data = torch.load(data_path)
        self.data_a = data['x_a']
    
    def __getitem__(self, index):
        return (self.data_a[index], ), -1

    def __len__(self):
        return len(self.data_a)

class toy_sin_b():
    def __init__(self, root='./data'):
        data_path = os.path.join(root, 'toy_sin', 'data.pt')
        data = torch.load(data_path)
        self.data_b = data['x_b']
    
    def __getitem__(self, index):
        return (self.data_b[index], ), -1

    def __len__(self):
        return len(self.data_b)