from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

import torch
import numpy as np
import pandas as pd
import os


class ReducedFeatureDataset(Dataset):
    def __init__(self, name="processed_dataset"):
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        self.TRAIN_PATH = os.path.join(current_dir, "data", "original", "train.csv")
        self.TEST_PATH = os.path.join(current_dir, "data", "original", "test.csv")
        self.dateset_name = name

    def get_numpy_data(self):
        return (self.data.numpy(), self.label.numpy(), self.submit_data.numpy())

    def get_test_id(self):
        return self.test_id

    def set_raw_label(self, raw_label):
        # pandas
        self.raw_label = raw_label

    def get_raw_label(self):
        # pandas
        return self.raw_label

    def convert_np_to_torch(self, data):
        return torch.tensor(data.values.astype(np.float32))

    def convert_oneshot(self, data, name):
        encoded_data = pd.get_dummies(
            data,
            columns=[
                name,
            ],
        )
        return encoded_data

    def transform_label(self):
        unique = self.raw_label.unique()
        self.le = LabelEncoder()
        raw_np_label = self.le.fit_transform(self.raw_label)
        self.label = torch.from_numpy(raw_np_label)
        self.label_number = unique.shape[0]

    def load(self):
        train = pd.read_table(self.TRAIN_PATH, delimiter="\t")
        test = pd.read_table(self.TEST_PATH, delimiter="\t")

        self.test_id = test["id"]
        self.raw_label = train["activity"]
        train_onehot_enc = pd.get_dummies(train["activity"])
        scaler = preprocessing.StandardScaler()
        tmp_train = train.drop(["id", "activity"], axis=1)
        train_scaled = pd.DataFrame(
            scaler.fit_transform(tmp_train),
            columns=tmp_train.columns,
            index=train.index,
        )
        test_scaled = pd.DataFrame(
            scaler.transform(test.drop(["id"], axis=1)),
            columns=tmp_train.columns,
            index=test.index,
        )

        self.data = self.convert_np_to_torch(train_scaled)
        self.label = self.convert_np_to_torch(train_onehot_enc)
        self.label_number = self.label.shape[1]
        self.submit_data = self.convert_np_to_torch(test_scaled)

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
