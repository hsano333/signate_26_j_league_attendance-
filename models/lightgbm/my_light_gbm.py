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

import lightgbm as lgb
from sklearn import datasets, preprocessing
from sklearn.metrics import classification_report

# from sklearn.metrics import roc_auc_score
from numpy import argmax
from common.utility import Utility
import optuna


class MyLightGBM(IModel):
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
        self.is_optimize = self.config.get("is_optimize", False)
        self.optuna_trial = optuna_trial

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

        # self.param = {
        #     "objective": self.config["objective"],
        #     "num_class": self.config["label_number"],
        #     "metric": self.config["metric"],
        #     "verbosity": self.config["verbosity"],
        #     "learning_rate": self.config["learning_rate"],
        #     "num_leaves": self.config["num_leaves"],
        #     "feature_fraction": self.config["feature_fraction"],
        #     "bagging_fraction": self.config["bagging_fraction"],
        #     "bagging_freq": self.config["bagging_freq"],
        #     "max_depth": self.config["max_depth"],
        #     "n_estimators": self.config["n_estimators"],
        # }

    @classmethod
    def get_name(self):
        current_dir = Path(__file__).resolve().parent.name
        return current_dir

    def learn(self, i, params=None):
        (train_data, train_label, _) = self.train_dataset.get_numpy_data()
        (val_data, val_label, _) = self.val_dataset.get_numpy_data()

        lgb_train = lgb.Dataset(train_data, label=train_label)
        lgb_val = lgb.Dataset(val_data, label=val_label)

        if self.model is not None:
            y_pred_proba = self.model.predict(val_data)
            y_pred = y_pred_proba.argmax(axis=1)
            score = self.evaluation(val_label, y_pred)
            print("lightgbm Model already exists. Skipping training.")
            print(f"Validation Accuracy: {score:.4f}")
            return

        if self.is_optimize and i == 0:
            callback = optuna.integration.LightGBMPruningCallback(
                self.optuna_trial, params["metric"], "valid"
            )
        else:
            callback = lgb.early_stopping(20)
        # callback = lgb.early_stopping(20)

        self.model = lgb.train(
            params,
            # params,
            # self.param,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=400,
            # early_stopping_rounds=50,
            valid_names=["valid"],
            # valid_names=["valid"],
            callbacks=[callback],
        )

        # 予測（各クラスの確率が返る）
        y_pred_proba = self.model.predict(val_data)
        # print(f"{y_pred_proba.shape=}")
        # y_pred = y_pred_proba > 0.5
        # y_pred = y_pred_proba.argmax(axis=1)

        # 評価
        score = self.evaluation(val_label, y_pred_proba)
        print(f"Validation Score: {score:.4f}")

        # 最適化のときはここで終了
        if self.is_optimize:
            return

        # print(f"{self.old_score=}")
        # print(f"{acc=}")
        # モデルの精度が前回よりも向上した場合のみ保存
        if score > self.old_score:
            data_to_save = {
                "model": self.model,
                "score": score,
            }
            # モデルを保存
            joblib.dump(data_to_save, self.save_model_path)
            # model_path = os.path.join(self.save_dir, "lightgbm_model.txt")
            # self.model.save_model(model_path)
            print(f"Model saved to {self.save_model_path}")
        else:
            print(
                f"Old Model Score:{self.old_score:.4f}\n No improvement in Score. Model not saved."
            )

    def reload(self):
        pass

    def forecast(self, data):
        # print(f"{self.save_model_path=}")
        # loaded_data = joblib.load(self.save_model_path)
        # self.model = loaded_data["model"]

        y_pred_proba = self.model.predict(data.numpy())
        return torch.tensor(y_pred_proba.astype(np.float32))

    def get_model_params(self, trial, model_config):
        return Utility().get_model_params(trial, model_config)

    def get_model_optimized_params(self, model_config):
        return model_config.get("learned_params", {})

    def check_params(self, params):
        pass
