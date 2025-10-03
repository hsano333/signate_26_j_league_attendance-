import my_ml
import sys
import numpy as np
from models.mytorch.my_torch import MyTorch
from ensemble.stacking import Stacking
from ensemble.blending import Blending
from pathlib import Path
import yaml
import torch
from common.utility import Utility

from models.xgboost.my_xgboost import MyXGBoost
from models.lightgbm.my_light_gbm import MyLightGBM
from models.randomforest.my_randomforest import MyRandomForest
from models.knn.my_knn import MyKNearestNeighbors
from models.linear_regression.my_linear_regression import MyLinearRegression
from models.svc.my_svc import MySupportVectorMachine
from models.logistic_regression.my_logistic_regression import MyLogisticRegression
from models.ransac.my_ransac import MyRANSAC
from models.polynomial.my_polynomial import MyPolynomial
from dataset.src.processed_dataset import ProcessedDataset
from dataset.src.time_dataset import TimeDataset
from dataset.src.processed_dataset_xgboost import ProcessedDatasetXGBoost
from dataset.src.processed_dataset_torch import ProcessedDatasetTorch
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

# from dataset.src.processed_dataset_dummies import ProcessedDatasetDummies
from optimizer.my_optuna import MyOptuna

from common.my_enum import MLTask
import pandas as pd
import copy

args = sys.argv
config = Utility().load_yaml_config()
task = MLTask.Regression
BaseEvaluation = root_mean_squared_error


def make_models():
    dataset = ProcessedDataset(config["original_dataset"], "jleag")
    dataset.load()

    dataset_xgboost = ProcessedDatasetXGBoost(
        config["xgboost_dataset"], "xgboost_titani"
    )
    dataset_xgboost.load()

    dataset_torch = ProcessedDatasetTorch(config["original_dataset"], "jleag_torch")
    dataset_torch.load()

    dataset_category = copy.deepcopy(dataset)
    dataset_category.transform_label(task)
    dataset_category_xgboost = copy.deepcopy(dataset_xgboost)
    dataset_category_xgboost.transform_label(task)
    my_torch = MyTorch
    #
    mytorch = ["mytorch", dataset_category_xgboost, my_torch, BaseEvaluation]
    time_random_forest = [
        "randomforest",
        dataset,
        MyRandomForest,
        BaseEvaluation,
    ]
    random_forest = [
        "randomforest",
        dataset_category_xgboost,
        MyRandomForest,
        BaseEvaluation,
    ]
    light_gbm = [
        "lightgbm",
        dataset_category_xgboost,
        MyLightGBM,
        BaseEvaluation,
    ]
    xgboost = ["xgboost", dataset_category_xgboost, MyXGBoost, BaseEvaluation]
    logistic_regression = [
        "logistic_regression",
        dataset_category_xgboost,
        MyLogisticRegression,
        BaseEvaluation,
    ]
    # poly = ["poly", time_dataset_category, time_model, BaseEvaluation]
    # poly = ["poly", time_dataset_category, MyPolynomial, BaseEvaluation]
    # time_xgboost = ["time_xgboost", time_dataset_category, time_model, BaseEvaluation]
    # time_xgboost = ["time_xgboost", time_dataset_category, time_model, BaseEvaluation]
    # time_xgboost = ["time_xgboost", time_dataset_category, MyXGBoost, BaseEvaluation]
    knn = ["knn", dataset_category, MyKNearestNeighbors, BaseEvaluation]
    svc = ["svc", dataset_category, MySupportVectorMachine, BaseEvaluation]
    models = {
        "mytorch": mytorch,
        "randomforest": random_forest,
        "lightgbm": light_gbm,
        "xgboost": xgboost,
        "logistic_regression": logistic_regression,
        "knn": knn,
        "svc": svc,
        # "poly": poly,
        # "time_xgboost": time_xgboost,
        # "time_linear_regression": time_model,
    }

    return models


def make_predict_models():
    logistic_regression = [
        "result_logistic_regression",
        None,
        MyLogisticRegression,
        BaseEvaluation,
    ]
    xgboost = ["result_xgboost", None, MyXGBoost, BaseEvaluation]
    linear_regression = [
        "result_linear_regression",
        None,
        MyLinearRegression,
        BaseEvaluation,
    ]
    models = {
        "xgboost": xgboost,
        "logistic_regression": logistic_regression,
        "linear_regression": linear_regression,
    }
    return models


def get_train_models(models):
    selected_models = []
    names = [
        # "mytorch",
        "randomforest",
        # "xgboost",
        # "lightgbm",
        # "knn",
        # "ransac",
        # "poly",
        # "time_xgboost",
        # "logistic_regression",
        # "svc",
        # "time_linear_regression",
    ]
    for name in names:
        selected_models.append(models[name])
    return selected_models


def get_predict_model(models):
    # selected_model = models["logistic_regression"]
    # selected_model = models["xgboost"]
    selected_model = models["linear_regression"]
    return selected_model


# その他必要なモデルをpredictに使用する場合に指定
# def get_other_models(models):
#     selected_models = []
#     selected_models.append(models["poly"])
#     return selected_models


def main():
    all_models = make_models()
    predict_models = make_predict_models()
    train_models = get_train_models(all_models)
    predict_model = get_predict_model(predict_models)
    # other_models = get_other_models(all_models)
    ensemble = Stacking(config, BaseEvaluation, task)
    # time_models = all_models["time_xgboost"]
    # ensemble = Blending(config, MLTask.Classification, False)

    if args[1] == "optimize":
        my_ml.optimize(train_models, task)
    elif args[1] == "train":
        my_ml.train(ensemble, train_models)
    elif args[1] == "optimize_predict":
        my_ml.optimize_predict(ensemble, train_models, predict_model, task, None)
    elif args[1] == "predict":
        my_ml.predict(ensemble, train_models, predict_model, BaseEvaluation, None)
    else:
        print(
            "Invalid argument. Use 'optimize', 'optimize_predict', 'train', or 'predict'."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
