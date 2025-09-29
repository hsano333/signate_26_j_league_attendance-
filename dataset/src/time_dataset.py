# from enum import auto
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dataset.src.simple_dataset import SimpleDataset
import joblib

import torch
import numpy as np
import pandas as pd
import os


class TimeDataset(Dataset):
    def __init__(self, config, time_data_model, name="processed_dataset"):
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        self.TRAIN_PATH = os.path.join(current_dir, "data", "original", "train.csv")
        self.TEST_PATH = os.path.join(current_dir, "data", "original", "test.csv")
        self.PREPOCESSED_PATH = os.path.join(
            current_dir, "data", "original", "dataset_preprocessed.pkl"
        )

        self.dateset_name = name
        self.config = config
        self.target_feature = self.config["target_feature"]

    def get_numpy_data(self):
        return (self.data.numpy(), self.label.numpy(), self.test_data.numpy())

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

    def convert_onehot_encoder(self, encoder, data, name):
        encoderd_data = encoder.transform(data[[name]])
        feature_names = encoder.get_feature_names_out([name])
        tmp_df = pd.DataFrame(encoderd_data, columns=feature_names)
        return pd.concat([data, tmp_df], axis=1)

    def convert_oneshot(self, data, name):
        encoded_data = pd.get_dummies(
            data,
            columns=[
                name,
            ],
        )
        return encoded_data

    def transform_label(self):
        pass

    def make_date_id(self):
        start_date = "2013-10-28"
        end_date = "2014-12-01"

        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame(all_dates, columns=["date"])
        df["datetime"] = pd.to_datetime(df["date"])
        df["weekday"] = df["datetime"].dt.day_name()
        # df["weekday_i"] = df["date"].dt.dayofweek

        df = df.reset_index()
        df = df.rename(columns={"index": "id"})
        df["week_id"] = df["id"] // 7
        # scaler = StandardScaler()
        # scaler.fit(df[["week_id"]])
        # df["week_id"] = scaler.transform(df[["week_id"]])
        df = df.drop(columns=["id", "date"])

        return df

    def get_mean_y(self, data, onehot_encoder):
        def trimmed_mean(x):
            if len(x) >= 3:
                return x.sort_values().iloc[0:-1].mean()
            else:
                return x.mean()  # データ数が2以下なら普通に平均

        # data["y_roll"] = data["y"].rolling(window=10, min_periods=1).mean()
        data["y_roll"] = data["y"]
        mean_y = data.groupby("week_id")["y_roll"].transform(trimmed_mean).reset_index()
        data["y"] = mean_y["y_roll"]

    def mypreprocessing(self, data):
        # tmp_data = time_liner_model.predict(data[["week_id"]])
        if "y" in data.columns:
            return data[["y", "week_id"]].astype(float)
        else:
            return data[["week_id"]].astype(float)

    def load(self):
        if os.path.exists(self.PREPOCESSED_PATH) and self.config["overwrite"] is False:
            print(f"Loading preprocessed dataset from {self.PREPOCESSED_PATH}")
            loaded_data = joblib.load(self.PREPOCESSED_PATH)
            self.data = loaded_data["data"]
            self.label = loaded_data["label"]
            self.label_number = self.label.shape[1]
            self.test_data = loaded_data["test_data"]
            self.test_id = loaded_data["datetime"]
            self.raw_label = loaded_data["raw_label"]
            self.target_feature = loaded_data["target_feature"]
            return

        train = pd.read_csv(self.TRAIN_PATH)
        test = pd.read_csv(self.TEST_PATH)

        self.test_id = test["datetime"]
        self.raw_label = train[self.target_feature]

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        onehot_encoder = ohe.fit(train[["name"]])

        train["datetime"] = pd.to_datetime(train["datetime"])
        test["datetime"] = pd.to_datetime(test["datetime"])
        date_df = self.make_date_id()
        train = pd.merge(train, date_df, on="datetime", how="left")
        test = pd.merge(test, date_df, on="datetime", how="left")

        self.get_mean_y(train, onehot_encoder)
        train = train.drop(["y_roll"], axis=1)

        train = self.mypreprocessing(train)
        test = self.mypreprocessing(test)

        tmp_label = train[self.target_feature]
        train = train.drop([self.target_feature], axis=1)

        # train_test = pd.concat([train, test], axis=0, ignore_index=True)
        train_scaled = pd.DataFrame(
            # scaler.fit_transform(train),
            train,
            columns=train.columns,
            index=train.index,
        )
        test_scaled = pd.DataFrame(
            # scaler.transform(test),
            test,
            columns=test.columns,
            index=test.index,
        )

        self.data = self.convert_np_to_torch(train_scaled)
        self.label = self.convert_np_to_torch(tmp_label).unsqueeze(
            1
        )  # Add a dimension for label
        # self.label_number = self.label.shape[1]
        self.label_number = 1
        self.test_data = self.convert_np_to_torch(test_scaled)

        data_to_save = {
            "data": self.data,
            "label": self.label,
            "test_data": self.test_data,
            "id": self.test_id,
            "raw_label": self.raw_label,
            "target_feature": self.target_feature,
        }
        joblib.dump(data_to_save, self.PREPOCESSED_PATH)
        print(f"Preprocessed dataset saved to {self.PREPOCESSED_PATH}")

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
