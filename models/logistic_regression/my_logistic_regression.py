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
import optuna

# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, preprocessing
from sklearn.metrics import classification_report

# from sklearn.metrics import roc_auc_score
from common.utility import Utility
from numpy import argmax


class MyLogisticRegression(IModel):
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
        # print(f"{params=}")
        (train_data, train_label, _) = self.train_dataset.get_numpy_data()
        (val_data, val_label, _) = self.val_dataset.get_numpy_data()

        if self.model is not None:
            y_pred_proba = self.model.predict_proba(val_data)[:, 1]
            score = self.evaluation(val_label, y_pred_proba)
            print("Logistic Regression Model already exists. Skipping training.")
            print(f"Validation Score: {score:.4f}")
            return

        # print(f"{params=}")
        logistic = LogisticRegression(
            **params,
            # random_state=self.random_state,
            # n_jobs=self.n_jobs,
            # n_estimators=self.n_estimators,
        )
        # print(f"{params=}")
        # print(f"{train_data.shape=}, {train_label.shape=}")
        # print(f"{train_data[0]=}, {train_label[0]=}")
        self.model = logistic.fit(train_data, train_label)

        y_pred = self.model.predict_proba(val_data)[:, 1]

        # 評価
        score = self.evaluation(val_label, y_pred)
        print(f"Logistic Regression Validation Score: {score:.4f}")

        # 最適化のときはここで終了
        if self.is_optimize:
            return

        if score > self.old_score:
            data_to_save = {
                "model": self.model,
                "score": score,
            }
            joblib.dump(data_to_save, self.save_model_path)
            # モデルを保存
            # model_path = os.path.join(self.save_dir, "lightgbm_model.txt")
            # self.model.save_model(model_path)
            print(f"Model saved to {self.save_model_path}")
        else:
            print(
                f"Old Model Score:{self.old_score:.4f}\n No improvement in Score. Model not saved."
            )

    def forecast(self, data):
        y_pred_proba = self.model.predict_proba(data.numpy())[:, 1]
        return torch.tensor(y_pred_proba.astype(np.float32))

    # def get_model_params(self, trial, model_config):
    #     params = {}
    #     config_opt = model_config["optimize"]
    #     solvers = config_opt["categorical"]["solver"]
    #
    #     params["solver"] = trial.suggest_categorical("solver", solvers)
    #     if (
    #         params["solver"] == "lbfgs"
    #         or params["solver"] == "newton-cg"
    #         or params["solver"] == "newton-cholesky"
    #         or params["solver"] == "sag"
    #     ):
    #         # 'lbfgs'は'l1', 'elasticnet'に対応していないため、これらを除外
    #         penalty = trial.suggest_categorical("penalty", ["l2", "none"])
    #         if penalty != "none":
    #             params["penalty"] = penalty
    #         else:
    #             params["penalty"] = None
    #     elif params["solver"] == "liblinear":
    #         # 'liblinear'は'l1', 'l2'に対応
    #         params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
    #     elif params["solver"] == "saga":
    #         # 'saga'はすべてのpenaltyに対応
    #         penalty = trial.suggest_categorical(
    #             "penalty", ["l1", "l2", "elasticnet", "none"]
    #         )
    #         if penalty != "none":
    #             params["penalty"] = penalty
    #         else:
    #             params["penalty"] = None
    #
    #     if "penalty" in params and params["penalty"] == "elasticnet":
    #         params = Utility().get_float_params(params, config_opt, trial)
    #     else:
    #         params = Utility().get_float_params(params, config_opt, trial, ["l1_ratio"])
    #
    #     params = Utility().get_int_params(params, config_opt, trial)
    #     params = Utility().get_categorical_params(
    #         params, config_opt, trial, ["solver", "penalty"]
    #     )
    #     params = Utility().get_const_params(params, config_opt, trial)
    #     print(f"get_model_params:{params=}")
    #     return params

    ### Optunaのパラメータで、エラーになるパラメーターを変更する
    def transform_params(self, trial, params):
        # print(f"pre:{params=}")
        # params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        if (
            params["solver"] == "lbfgs"
            or params["solver"] == "newton-cg"
            or params["solver"] == "newton-cholesky"
            or params["solver"] == "sag"
        ):
            # params["penalty"] = trial.suggest_categorical("penalty", ["l2", "none"])
            params["penalty"] = trial.suggest_categorical("penalty", ["l2"])
        elif params["solver"] == "liblinear":
            params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        elif params["solver"] == "saga":
            params["penalty"] = trial.suggest_categorical(
                "penalty", ["l1", "l2", "elasticnet"]
            )

        if params["penalty"] == "l1" and params["solver"] in [
            "lbfgs",
            "newton-cg",
            "newton-cholesky",
            "sag",
        ]:
            params["solver"] = trial.suggest_categorical(
                "solver", ["liblinear", "saga"]
            )
        if params["penalty"] == "none" and params["solver"] == "liblinear":
            params["solver"] = trial.suggest_categorical(
                "solver", ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]
            )
        if params["penalty"] == "elasticnet":
            params["solver"] = "saga"
        else:
            params["l1_ratio"] = None
            # params.pop("l1_ratio", None)

        # if params["class_weight"] == "balanced":
        # print(f"post:{params=}")

        return params

    def get_model_params(self, trial, model_config):
        params = Utility().get_model_params(trial, model_config)
        return params
        # if params["penalty"] == "none":
        #     params["penalty"] = None

    def get_model_optimized_params(self, model_config):
        params = model_config.get("learned_params", {})
        if params.get("penalty") == "none":
            params["penalty"] = None
        return params

    def check_params(self, params):
        valid_combos = {
            "none": ["newton-cg", "lbfgs", "sag"],
            "l1": ["liblinear", "saga"],
            "l2": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "elasticnet": ["saga"],
        }

        # print(f"check_params:{params=}")
        penalty = "none" if params["penalty"] is None else params["penalty"]
        # print(f"{penalty=}")
        solver = params["solver"]
        if solver not in valid_combos[penalty]:
            raise optuna.TrialPruned()

        # penaltyがelasticnet以外ならl1_ratioは固定値にして無視
        if penalty != "elasticnet":
            l1_ratio = None
            pass
