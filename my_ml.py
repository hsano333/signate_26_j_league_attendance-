import sys
from ensemble.stacking import Stacking
from models.mytorch.my_torch import MyTorch
from pathlib import Path
import yaml
from common.utility import Utility

from models.xgboost.my_xgboost import MyXGBoost
from models.lightgbm.my_light_gbm import MyLightGBM
from models.randomforest.my_randomforest import MyRandomForest
from models.knn.my_knn import MyKNearestNeighbors
from models.svc.my_svc import MySupportVectorMachine
from models.logistic_regression.my_logistic_regression import MyLogisticRegression
from dataset.src.processed_dataset import ProcessedDataset
from dataset.src.time_dataset import TimeDataset
from dataset.src.processed_dataset_xgboost import ProcessedDatasetXGBoost
from dataset.src.processed_dataset_dummies import ProcessedDatasetDummies
from optimizer.my_optuna import MyOptuna

from common.my_enum import MLTask
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import torch

# ConfigFileName = "config.yaml"
# config_path = Path(__file__).parent.resolve() / "common" / ConfigFileName
config = Utility().load_yaml_config()


def optimize(models, task=MLTask.Classification):
    optimizer = MyOptuna(config, task)
    for model in models:
        optimizer.add_model(*model)
    optimizer.optimize_models(task, True)


def optimize_predict(ensemble, train_models, predict_model, task, other_models):
    for model in train_models:
        ensemble.add_model(*model)

    tmp_dataset = train_models[0][1]
    raw_label = tmp_dataset.get_raw_label()

    result_dataset = ensemble.make_features(forece_make=False)
    result_dataset.set_raw_label(raw_label)
    # result_dataset.transform_label()

    optimizer = MyOptuna(config, task)
    predict_model[1] = result_dataset
    optimizer.add_model(*predict_model)
    optimizer.optimize_models(task, True)

    ensemble.clear_models()
    predict_model[1] = result_dataset
    ensemble.add_model(*predict_model)
    result_dataset = ensemble.make_features(forece_make=False)


def train(ensemble, models):
    # ensemble = Stacking(config, MLTask.Classification)
    for model in models:
        ensemble.add_model(*model)
    params = {"forece_make": True}
    ensemble.learn(0, params)
    # result_dataset = ensemble.make_features(forece_make=True)


def predict(ensemble, train_models, predict_model, evaluation, other_models):
    for model in train_models:
        ensemble.add_model(*model)

    dataset = train_models[0][1]
    label_scaler = dataset.get_label_scaler()
    # (train_offset, test_offset) = dataset.get_mean_df()

    # raw_label_data = label_scaler.inverse_transform(dataset.get_raw_label().to_frame())
    raw_label_data = dataset.get_raw_label()
    test_data = dataset.get_numpy_data()[2]

    result_dataset = ensemble.make_features(forece_make=False)
    result_dataset.set_raw_label(dataset.get_raw_label())

    ensemble.clear_models()
    predict_model[1] = result_dataset
    ensemble.add_model(*predict_model)

    final_result = ensemble.forecast_and_get_data()
    pred_data = label_scaler.inverse_transform(final_result.get_numpy_data()[0])
    label = final_result.get_numpy_data()[1]

    # print(f"{type(train_offset)=}")
    # print(f"{type(pred_data)=}")
    # print(f"{(train_offset.shape)=}")
    # print(f"{(pred_data.shape)=}")
    # train_score = evaluation(raw_label_data, pred_data + train_offset.to_frame())
    train_score = evaluation(raw_label_data, pred_data)
    # train_score = evaluation(raw_label_data, pred_data)
    print(f"Result Train Result Score: {train_score:.4f}")

    test_data = (
        label_scaler.inverse_transform(final_result.get_numpy_data()[2])
        # + test_offset.to_frame().to_numpy()
    )
    print(f"{type(test_data)=}")

    id = dataset.get_test_id()
    df = pd.DataFrame({"id": id, "activity": test_data.reshape(-1)})
    df.to_csv("submission.csv", index=False, header=False, sep=",")

    # df2 = pd.DataFrame({"activity": pred_data.reshape(-1)})
    # df2.to_csv("submission_pred.tsv", index=False, header=False, sep="\t")
