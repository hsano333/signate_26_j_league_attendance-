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

import xgboost as xgb
from sklearn import datasets, preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from numpy import argmax
from common.utility import Utility

BATCH_SIZE = 40
EPOCH_SIZE = 100
NUM_WORKERS = 4


class MyXGBoost(IModel):
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
        if save_dir is None:
            self.save_dir = Path(__file__).resolve().parent
        self.save_model_path = os.path.join(
            self.save_dir, self.config["model_file_name"]
        )
        self.old_score = 0.0
        self.is_optimize = self.config.get("is_optimize", False)
        self.n_estimators = 150
        self.proba = config.get("proba", True)

        # if "multi" in self.config["objective"]:
        #     self.param = {
        #         "objective": config["objective"],
        #         "eval_metric": config["eval_metric"],
        #         "learning_rate": config["learning_rate"],
        #         "num_class": config["num_class"],
        #         "max_depth": config["max_depth"],
        #         "subsample": config["subsample"],
        #         "n_estimators": config["n_estimators"],
        #         "colsample_bytree": config["colsample_bytree"],
        #         "min_child_weight": config["min_child_weight"],
        #         # "num_class": config["label_number"],
        #     }
        # elif "binary" in self.config["objective"]:
        #     self.param = {
        #         "objective": config["objective"],
        #         "eval_metric": config["eval_metric"],
        #         "learning_rate": config["learning_rate"],
        #     }
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

    def learn(self, i, params=None):
        (train_data, train_label, _) = self.train_dataset.get_numpy_data()
        # print(
        #     f"train_data.shape: {train_data.shape}, train_label.shape: {train_label.shape}"
        # )
        (val_data, val_label, _) = self.val_dataset.get_numpy_data()
        # (features, label, _) = self.train_dataset.get_numpy_data()
        xgb_train = xgb.DMatrix(train_data, label=train_label)
        xgb_val = xgb.DMatrix(val_data, label=val_label)
        if self.model is not None:
            if "multi" in self.config["objective"]:
                y_pred = argmax(self.model.predict(val_data), axis=1)
            elif "binary" in self.config["objective"]:
                # y_pred_proba = self.model.predict(val_data)
                # y_pred = (y_pred_proba >= 0.5).astype(int)
                y_pred = self.model.predict(val_data)

            # y_pred = y_pred_proba.argmax(axis=1)
            score = self.evaluation(val_label, y_pred)
            print("Xgboost Model already exists. Skipping training.")
            print(f"Validation Score: {score:.4f}")
            return

        if "n_estimators" in params:
            self.n_estimators = params.pop("n_estimators")
        self.model = xgb.train(params, xgb_train, num_boost_round=self.n_estimators)

        if "multi" in self.config["objective"]:
            y_pred = argmax(self.model.predict(xgb_train), axis=1)
        elif "binary" in self.config["objective"]:
            # y_proba = self.model.predict(xgb_train)
            # y_pred = (y_proba >= 0.5).astype(int)
            y_pred = self.model.predict(xgb_train)
        elif "reg:squarederror" in self.config["objective"]:
            y_pred = self.model.predict(xgb_train)

        # 最適化のときはここで終了
        if self.is_optimize:
            return

        # predictions = (y_
        # print(f"XGBoost Predictions: {y_proba=}, {y_pred=}, {train_label[:10]=}")
        # print(f"{train_label=}, {y_pred=}")
        score = self.evaluation(train_label, y_pred)
        print(f"XGBoost Score: {score:.4f}")
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

    def reload(self):
        pass

    def forecast(self, data):
        xgb_forecast = xgb.DMatrix(data.numpy())
        if "multi" in self.config["objective"]:
            predictions = argmax(self.model.predict(xgb_forecast), axis=1)
        elif "binary" in self.config["objective"]:
            predictions = self.model.predict(xgb_forecast)
            predictions = predictions.reshape(-1, 1)
        elif "reg:squarederror" in self.config["objective"]:
            predictions = self.model.predict(xgb_forecast)
            predictions = predictions.reshape(-1, 1)

        return torch.tensor(predictions.astype(np.float32))

    def get_model_params(self, trial, model_config):
        params = Utility().get_model_params(trial, model_config)
        # histの計算をCUDA(GPU)で行うが、実際やるとCPUより遅かった。
        # データサイズが小さいのが原因？
        # if params.get("tree_method") == "hist":
        #     params["device"] = "cuda"
        return params

    def get_model_optimized_params(self, model_config):
        return model_config.get("learned_params", {})

    def check_params(self, params):
        pass
