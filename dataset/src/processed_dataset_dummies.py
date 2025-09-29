# from enum import auto
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import joblib

import torch
import numpy as np
import pandas as pd
import os


class ProcessedDatasetDummies(Dataset):
    def __init__(self, config, name="processed_dataset"):
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

    def preprocessing(self, data, scaler, day_ratio):
        data = data.merge(
            day_ratio[["day", "month", "day_bin"]], on=["day", "month"], how="left"
        )
        data["day_bin"] = data["day_bin"].fillna(0).astype(int)

        data["duration"] = pd.to_numeric(data["duration"])
        data["pdays"] = pd.to_numeric(data["pdays"])
        data["balance"] = pd.to_numeric(data["balance"])
        data["campaign"] = pd.to_numeric(data["campaign"])
        data["previous"] = pd.to_numeric(data["previous"])
        data.loc[data["pdays"] >= 0, "pdays_dummy"] = "1"
        data.loc[~data["pdays"] >= 0, "pdays_dummy"] = "0"
        data.loc[data["balance"] < 0, "balance_dummy"] = "-1"
        data.loc[data["balance"] == 0, "balance_dummy"] = "zero"
        data.loc[data["balance"] >= 0, "balance_dummy"] = "1"
        data.loc[data["balance"] >= 2000, "balance_dummy"] = "2"
        data.loc[data["balance"] >= 13000, "balance_dummy"] = "3"
        data.loc[data["balance"] >= 35000, "balance_dummy"] = "4"

        data.loc[data["campaign"] == 0, "campaign_dummy"] = "0"
        data.loc[data["campaign"] >= 1, "campaign_dummy"] = "1"
        data.loc[data["campaign"] >= 4, "campaign_dummy"] = "2"
        data.loc[data["campaign"] >= 9, "campaign_dummy"] = "3"
        data.loc[data["campaign"] >= 18, "campaign_dummy"] = "4"
        data.loc[data["campaign"] >= 25, "campaign_dummy"] = "5"

        data["poutcome_dummy"] = "0"
        data.loc[data["poutcome"] == "success", "poutcome_dummy"] = "1"

        # data["day_dummy"] = "0"
        # data.loc[data["day"] == 1, "day_dummy"] = "1"
        # data.loc[data["day"] == 10, "day_dummy"] = "1"
        # data.loc[data["day"] == 3, "day_dummy"] = "2"
        # data.loc[data["day"] == 4, "day_dummy"] = "2"
        # data.loc[data["day"] == 22, "day_dummy"] = "2"
        # data.loc[data["day"] == 24, "day_dummy"] = "2"
        # data.loc[data["day"] == 30, "day_dummy"] = "2"
        # data.loc[data["day"] == 2, "day_dummy"] = "3"
        # data.loc[data["day"] == 2, "day_dummy"] = "3"
        # data.loc[data["day"] == 11, "day_dummy"] = "3"
        # data.loc[data["day"] == 12, "day_dummy"] = "3"
        # data.loc[data["day"] == 13, "day_dummy"] = "3"
        # data.loc[data["day"] == 14, "day_dummy"] = "3"
        # data.loc[data["day"] == 15, "day_dummy"] = "3"
        # data.loc[data["day"] == 16, "day_dummy"] = "3"
        # data.loc[data["day"] == 23, "day_dummy"] = "3"
        # data.loc[data["day"] == 25, "day_dummy"] = "3"
        # data.loc[data["day"] == 26, "day_dummy"] = "3"
        # data.loc[data["day"] == 27, "day_dummy"] = "3"

        data["age_dummy"] = "0"
        data.loc[data["age"] >= 18, "age_dummy"] = "1"
        data.loc[data["age"] >= 23, "age_dummy"] = "2"
        data.loc[data["age"] >= 26, "age_dummy"] = "3"
        data.loc[data["age"] >= 30, "age_dummy"] = "4"
        data.loc[data["age"] >= 45, "age_dummy"] = "5"
        data.loc[data["age"] == 60, "age_dummy"] = "6"
        data.loc[data["age"] >= 61, "age_dummy"] = "7"
        data.loc[data["age"] >= 95, "age_dummy"] = "8"

        for i in range(0, 2100, 100):
            data.loc[
                (data["duration"] >= i) & (data["duration"] < i + 100), "duration_dummy"
            ] = str(i)
        data.loc[data["duration"] >= 2100, "duration_dummy"] = "2100"

        data.loc[data["previous"] == 0, "previous_dummy"] = "0"
        data.loc[data["previous"] >= 1, "previous_dummy"] = "1"
        data.loc[data["previous"] >= 3, "previous_dummy"] = "2"
        data.loc[data["previous"] >= 5, "previous_dummy"] = "3"
        data.loc[data["previous"] >= 8, "previous_dummy"] = "4"
        data.loc[data["previous"] >= 15, "previous_dummy"] = "5"

        data = self.convert_oneshot(data, "default")
        data = self.convert_oneshot(data, "job")
        data = self.convert_oneshot(data, "marital")
        data = self.convert_oneshot(data, "education")
        data = self.convert_oneshot(data, "housing")
        data = self.convert_oneshot(data, "loan")
        data = self.convert_oneshot(data, "pdays_dummy")
        data = self.convert_oneshot(data, "poutcome_dummy")
        data = self.convert_oneshot(data, "balance_dummy")
        data = self.convert_oneshot(data, "campaign_dummy")
        # data = self.convert_oneshot(data, "day_dummy")
        data = self.convert_oneshot(data, "age_dummy")
        data = self.convert_oneshot(data, "duration_dummy")
        data = self.convert_oneshot(data, "previous_dummy")
        data = self.convert_oneshot(data, "day_bin")

        data = data.drop(
            [
                "id",
                "day",
                "month",
                "poutcome",
                "contact",
                "pdays",
                "poutcome",
                "balance",
                "campaign",
                "age",
                "duration",
                "previous",
                # "y_true_ratio",
            ],
            axis=1,
        )
        # print(f"test_test {data.shape=}, {data[0:]=}")
        return data.astype(float)

    def load(self):
        if os.path.exists(self.PREPOCESSED_PATH) and self.config["overwrite"] is False:
            print(f"Loading preprocessed dataset from {self.PREPOCESSED_PATH}")
            loaded_data = joblib.load(self.PREPOCESSED_PATH)
            self.data = loaded_data["data"]
            self.label = loaded_data["label"]
            self.label_number = self.label.shape[1]
            self.test_data = loaded_data["test_data"]
            self.test_id = loaded_data["id"]
            self.raw_label = loaded_data["raw_label"]
            self.target_feature = loaded_data["target_feature"]
            return

        scaler = preprocessing.StandardScaler()
        train = pd.read_table(self.TRAIN_PATH, delimiter="\t")
        test = pd.read_table(self.TEST_PATH, delimiter="\t")
        self.test_id = test["id"]
        train_id = train["id"]
        test_id = test["id"]

        train_test = pd.concat([train, test], axis=0, ignore_index=True)

        # train_test["embarked"].fillna(("S"), inplace=True)
        train_test["age"] = train_test["age"].fillna(train_test["age"].mean())
        train_test["embarked"] = train_test["embarked"].fillna(
            train_test["embarked"].mode()
        )
        # train_test["sex"] = LabelEncoder().fit_transform(train_test["sex"])
        # train_test["embarked"] = LabelEncoder().fit_transform(
        #     train_test["embarked"].astype(str)
        # )
        # imputer = KNNImputer(n_neighbors=5)
        cols_to_impute = [
            "age",
            "pclass",
            "sex",
            "sibsp",
            "parch",
            "fare",
            "embarked",
        ]
        # imputer.fit(train_test[cols_to_impute])
        # train_test[cols_to_impute] = imputer.fit_transform(train_test[cols_to_impute])

        family_label = LabelEncoder()
        # train_test.loc[:, "Family"] = train_test["sibsp"] + train_test["parch"] + 1
        # train_test = pd.get_dummies(train_test)
        train = train_test.loc[train.index].drop(["id"], axis=1)
        test = train_test.loc[test.index].drop(["id"], axis=1)
        train = pd.get_dummies(train)
        test = pd.get_dummies(test)
        # print(train.isnull().sum())
        # print(test.isnull().sum())
        # family_label.fit(train_test["Family"].values)

        self.raw_label = train[self.target_feature]

        # scaler.fit(train_test[["age", "fare", "Family"]])

        # train = self.preprocessing(train, scaler, family_label, imputer)
        # test = self.preprocessing(test, scaler, family_label, imputer)
        tmp_label = train[self.target_feature]
        train = train.drop([self.target_feature], axis=1)
        test = test.drop([self.target_feature], axis=1)
        # print(f"{train.shape=}, {test.shape=}, {tmp_label.shape=}")
        # print(f"{train=}")
        # print(f"{test=}")

        # train_test = pd.concat([train, test], axis=0, ignore_index=True)
        train_scaled = pd.DataFrame(
            # scaler.fit_transform(train),
            train,
            columns=train.columns,
            index=train.index,
        )
        # print(f"{train_scaled.shape=}, {train_scaled[0:]=}")
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
        # print(f"{self.data.shape=}, {self.label.shape=}, {self.test_data.shape=}")
        # print(f"{self.data=}")
        # print(f"{self.label=}")

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
