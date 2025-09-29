from torch.utils.data import Dataset
import torch
import numpy as np


class SimpleDataset(Dataset):
    def __init__(self, data, label, label_number, name="combined"):
        super().__init__()
        if type(data) is np.ndarray:
            self.data = torch.from_numpy(data)
        else:
            self.data = data
        if type(label) is np.ndarray:
            self.label = torch.from_numpy(label)
        else:
            self.label = label
        self.name = name
        self.label_number = label_number

    def get_numpy_data(self):
        return (self.data.numpy(), self.label.numpy(), None)

    def load(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_name(self):
        return self.dateset_name

    def get_column_number(self):
        return self.data.shape[1]

    def get_label_number(self):
        return self.label_number
