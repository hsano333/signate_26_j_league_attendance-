from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dataset.src.simple_dataset import SimpleDataset
from common.my_enum import MLTask


class CombinedDataset(Dataset):
    def __init__(self, train_data, dataset, test_data, name="combined"):
        super().__init__()

        (_, train_label, _) = dataset.get_numpy_data()
        self.train_data = torch.from_numpy(train_data)
        self.train_label = torch.from_numpy(train_label)
        self.test = torch.from_numpy(test_data)
        self.dateset_name = name
        self.label_number = dataset.get_label_number()

    def get_numpy_data(self):
        return (self.train_data.numpy(), self.train_label.numpy(), self.test.numpy())

    def transform_label(self, task):
        if task == MLTask.Regression:
            self.label_number = self.label.shape[0]
        else:
            unique = self.raw_label.unique()
            self.le = LabelEncoder()
            # raw_np_label = self.le.fit_transform(self.raw_label)
            raw_np_label = self.le.fit_transform(self.label)
            self.train_label = torch.from_numpy(raw_np_label)
            self.label_number = unique.shape[0]

    def get_label_encoder(self):
        return self.le

    def set_raw_label(self, raw_label):
        # pandas
        self.raw_label = raw_label

    def get_raw_label(self):
        # pandas
        return self.raw_label

    def load(self):
        pass

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, ndx):
        return (self.train_data[ndx], self.train_label[ndx])

    def get_name(self):
        return self.dateset_name

    def get_label_number(self):
        return self.label_number

    def get_train_dataset(self):
        (train_data, train_label, test_data) = self.get_numpy_data()
        return SimpleDataset(
            train_data,
            train_label,
            0,
            self.dateset_name + "_train" + str(0 + 1),
        )

    def get_test_dataset(self):
        (train_data, train_label, test_data) = self.get_numpy_data()
        return SimpleDataset(
            test_data,
            None,
            0,
            self.dateset_name + "_train" + str(0 + 1),
        )
