import os
import importlib
import pandas as pd
import sys
from models.abstract_model import IModel
from pathlib import Path
import torch._dynamo
import time
import datetime
import numpy as np
import joblib

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import datasets, preprocessing
from sklearn.metrics import classification_report

# from sklearn.metrics import roc_auc_score
from numpy import argmax
from common.utility import Utility


class MyLinearRegression(IModel):
    def __init__(
        self,
        config,
        save_dir,
        train=None,
        val=None,
        test=None,
        evaluation=None,
        optuna_trial=None,
    ):
        self.model = None
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.evaluation = evaluation
        self.save_dir = save_dir
        self.save_model_path = os.path.join(
            self.save_dir, self.config["model_file_name"]
        )
        self.save_model_path = os.path.join(self.save_dir, self.save_model_path)
        # self.random_state = self.config["random_state"]
        self.n_jobs = -1
        # self.n_estimators = self.config["n_estimators"]
        self.is_optimize = self.config.get("is_optimize", False)

        self.old_score = 0.0
        if os.path.isdir(self.save_dir) is False:
            os.makedirs(self.save_dir)
        elif self.is_optimize:
            pass
        elif self.config["overwrite"] is True:
            if os.path.isfile(self.save_model_path):
                os.remove(self.save_model_path)
        elif os.path.isfile(self.save_model_path):
            loaded_data = joblib.load(self.save_model_path)
            self.model = loaded_data["model"]
            self.old_score = loaded_data["score"]

    @classmethod
    def get_name(self):
        current_dir = Path(__file__).resolve().parent.name
        return current_dir

    def reload(self):
        pass

    def learn(self, i, params=None):
        (train_data, train_label, _) = self.train_dataset.get_numpy_data()
        (val_data, val_label, _) = self.val_dataset.get_numpy_data()

        if self.model is not None:
            y_pred_proba = self.model.predict(val_data)[:, 1]
            score = self.evaluation(val_label, y_pred_proba)
            print("Linear Regression Model already exists. Skipping training.")
            print(f"Validation Score: {score:.4f}")
            return

        logistic = LinearRegression(
            # **params,
            # n_neighbors=self.config["n_neighbors"],
            # random_state=self.random_state,
            # n_jobs=self.n_jobs,
            # n_estimators=self.n_estimators,
        )
        min = params["weight_min"]
        max = params["weight_max"]
        weights = np.linspace(min, max, len(train_data))
        self.model = logistic.fit(train_data, train_label, sample_weight=weights)

        y_pred = self.model.predict(val_data)

        # 評価
        score = self.evaluation(val_label, y_pred)
        print(f"Linear Regression Validation Score: {score:.4f}")

        # 最適化のときはここで終了
        if self.is_optimize:
            return

        if score > self.old_score:
            data_to_save = {
                "model": self.model,
                "score": score,
            }
            # モデルを保存
            joblib.dump(data_to_save, self.save_model_path)
            print(f"Model saved to {self.save_model_path}")
        else:
            print(
                f"Old Model Score:{self.old_score:.4f}\n No improvement in Score. Model not saved."
            )

    def forecast(self, data):
        y_pred_proba = self.model.predict(data.numpy())
        return torch.tensor(y_pred_proba.astype(np.float32))

    def get_model_params(self, trial, model_config=None):
        return Utility().get_model_params(trial, self.config)

    def get_model_optimized_params(self, model_config=None):
        return self.config.get("learned_params", {})

    def check_params(self, params):
        pass
