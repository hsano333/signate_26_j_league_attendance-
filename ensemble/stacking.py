from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from models.abstract_model import IModel
from ensemble.oof import OOF
import shutil

from pathlib import Path
from common.utility import Utility

from dataset.src.combined_dataset import CombinedDataset
from dataset.src.simple_dataset import SimpleDataset
from common.my_enum import MLTask
import torch
import numpy as np
import os
import yaml


class Stacking(IModel):
    DefaultDirectory = "stacking_making_features"

    def __init__(self, config, evaluation, task=MLTask.Classification):
        self.name = self.get_name()
        self.config = config
        # self.dataset_models = {}
        self.dir_path = Path(__file__).parent.resolve() / self.DefaultDirectory
        # self.dir_path.mkdir(parents=True, exist_ok=True)
        self.KFoldSplit = self.config["stacking"]["k_fold"]
        self.RANDAOM_STATE = self.config["stacking"]["random_state"]
        self.task = task
        self.oof = OOF(self.name, config, self.dir_path, task)
        self.evaluation = evaluation
        # self.model_config = self.config["stacking"]["model_config"]
        # self.ForceMake = self.config["stacking"]["force_make"]
        # print(f"{self.KFoldSplit=}, {self.RANDAOM_STATE=}")

    @classmethod
    def get_name(self):
        return "stacking"
        # current_dir = Path(__file__).resolve().parent.name
        # return current_dir

    def reload(self):
        pass

    def delete_all_files(self, path):
        print(f"delete_all_files: {path=}")
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  # ファイルやシンボリックリンクを削除
            elif item.is_dir():
                shutil.rmtree(item)

    def add_model(self, name, dataset, model, evaluation):
        self.oof.add_model(name, dataset, model, evaluation)
        # self.dataset_models[name] = (dataset, model, evaluation)

    def clear_models(self):
        self.oof.dataset_models.clear()

    def forecast(self):
        final_result = self.forecast_and_get_data()
        # final_result = ensemble.forecast()
        pred_data = final_result.get_numpy_data()[0]
        label = final_result.get_numpy_data()[1]
        test_data = final_result.get_numpy_data()[2]

        train_score = self.evaluation(label, pred_data)
        print(f"Train Result Score: {train_score:.4f}")
        return test_data

    # def make_param(self, forece_make=False):
    #     # preds = []
    #     # tests = []
    #
    #     preds = np.zeros(len(self.dataset_models))
    #     tests = np.zeros(len(self.dataset_models))
    #     for name, (dataset, model, evaluation) in self.dataset_models.items():
    #         model_name = name
    #         dataset_name = dataset.get_name()
    #
    #         (directory, file_cv_trained_path, file_cv_new_test_path) = (
    #             self.make_directory(dataset_name, model_name)
    #         )
    #
    #         if not os.path.isfile(file_cv_trained_path) or forece_make:
    #             # directory.mkdir(parents=True, exist_ok=True)
    #             (pred, test) = self.make_new_feature(
    #                 model, model_name, dataset, directory, evaluation
    #             )
    #             np.savez(file_cv_trained_path, pred=pred, test=test)
    #             print("make:" + file_cv_trained_path)
    #         else:
    #             data = np.load(file_cv_trained_path)
    #             pred = data["pred"]
    #             test = data["test"]
    #             # preds[i] = data["pred"]
    #             # tests[i] = data["test"]
    #             print("load:" + file_cv_trained_path)
    #
    #         # preds.append(pred)
    #         # tests.append(test)
    #         preds[i] = data["pred"]
    #         tests[i] = data["test"]
    #
    #     result_pred = np.hstack(preds)
    #     result_test = np.hstack(tests)
    #     # ラベルはすべてのデータセットで同じなので、最初のデータセットから取得
    #     (_, label, _) = next(iter(self.dataset_models.values()))[0].get_numpy_data()
    #     np.savez(file_cv_new_test_path, pred=result_pred, test=result_test)
    #
    #     combined_dataset = CombinedDataset(
    #         result_pred, dataset, result_test, "stacking_combined"
    #     )
    #     return combined_dataset

    def learn(self, i, params=None):
        force_make = params.get("forece_make", False) if params else False
        self.oof.make_features(force_make)

    def make_features(self, forece_make=False):
        return self.oof.make_features(forece_make)

    #     preds = []
    #     tests = []
    #
    #     i = -1
    #     for name, (dataset, model, evaluation) in self.dataset_models.items():
    #         i = i + 1
    #         model_name = name
    #         dataset_name = dataset.get_name()
    #         (directory, file_cv_trained_path, file_cv_new_test_path) = (
    #             self.make_directory(dataset_name, model_name)
    #         )
    #
    #         if not os.path.isfile(file_cv_trained_path) or forece_make:
    #             directory.mkdir(parents=True, exist_ok=True)
    #             (pred, test) = self.make_new_feature(
    #                 model, model_name, dataset, directory, evaluation
    #             )
    #             np.savez(file_cv_trained_path, pred=pred, test=test)
    #             print("make:" + file_cv_trained_path)
    #         else:
    #             data = np.load(file_cv_trained_path)
    #             pred = data["pred"]
    #             test = data["test"]
    #             print("load:" + file_cv_trained_path)
    #
    #         preds.append(pred.reshape(-1, 1))
    #         tests.append(test.reshape(-1, 1))
    #
    #     result_pred = np.hstack(preds)
    #     result_test = np.hstack(tests)
    #
    #     # ラベルはすべてのデータセットで同じなので、最初のデータセットから取得
    #     (_, label, _) = next(iter(self.dataset_models.values()))[0].get_numpy_data()
    #     np.savez(file_cv_new_test_path, pred=result_pred, test=result_test)
    #
    #     combined_dataset = CombinedDataset(
    #         result_pred, dataset, result_test, "stacking_combined"
    #     )
    #     return combined_dataset

    # def fit(self):
    #     self.dataset.load()
    #     (train, label, test) = self.dataset.get_numpy_data()
    #     # print(f"{train.shape=}")
    #
    #     # Fit the model on the training data
    #     self.model.fit(train, label)

    # def predict(self):
    #     _, _, test = self.dataset.get_numpy_data()
    #     predictions = self.model.predict(test)
    #     return predictions

    def get_model_params(self, trial, model_config):
        return None

    def get_model_optimized_params(self, model_config):
        return None

    def check_params(self, params):
        pass

    def forecast_and_get_data(self) -> CombinedDataset:
        # def forecast(self) -> CombinedDataset:
        if len(self.oof.dataset_models) != 1:
            raise ValueError("OOF requires exactly one dataset and one model to learn.")

        name = list(self.oof.dataset_models.keys())[0]
        (dataset, model, evaluation, _) = self.oof.dataset_models[name]
        model_name = name
        dataset_name = dataset.get_name()

        name = dataset_name + "_" + model_name + "_final" + str(self.KFoldSplit)
        directory = self.dir_path / name
        file_cv_trained_path = (directory / "cv_trained.npz").as_posix()
        file_cv_new_test_path = (directory / "cv_new_test.npz").as_posix()
        if directory.exists() is False:
            directory.mkdir(parents=True, exist_ok=True)

        directory.mkdir(parents=True, exist_ok=True)
        (pred, test) = self.learn_final(
            model, model_name, dataset, directory, evaluation
        )
        # ラベルはすべてのデータセットで同じなので、最初のデータセットから取得
        (_, label, _) = next(iter(self.oof.dataset_models.values()))[0].get_numpy_data()
        np.savez(file_cv_new_test_path, pred=pred, test=test)

        combined_dataset = CombinedDataset(
            pred.numpy(), dataset, test.numpy(), self.name
        )
        return combined_dataset

    def learn_final(self, model, model_name, dataset, directory, evaluation):
        # kf = self.get_kf(task)
        (data, label, test) = dataset.get_numpy_data()
        label_number = dataset.get_label_number()

        tmp_name = dataset.get_name()
        preds = []
        preds_test = []
        va_idxes = []
        label_num = dataset.get_label_number()

        if label.ndim == 1:
            split_label = label
        else:
            split_label = label[:, 0]

        tr_idx = 0
        val_idx = 0
        i = 0
        # for i, (tr_idx, val_idx) in enumerate(kf.split(data, split_label)):
        k_directory = directory / f"fold_{i + 1}"
        va_idxes.append(val_idx)
        # train_data, val_data = data[tr_idx], data[val_idx]
        # train_label, val_label = label[tr_idx], label[val_idx]
        train_data, val_data = data, data
        train_label, val_label = label, label
        new_train = SimpleDataset(
            train_data,
            train_label,
            label_number,
            tmp_name + "_train" + str(i + 1),
        )
        new_val = SimpleDataset(
            val_data,
            val_label,
            label_number,
            tmp_name + "_val" + str(i + 1),
        )
        new_test = SimpleDataset(
            test,
            None,
            label_number,
            tmp_name + "_test" + str(i + 1),
        )

        model_config = self.config["model"][model_name]
        model_config["label_number"] = label_num
        model_config["is_optimize"] = False
        new_model = model(
            model_config,
            k_directory.as_posix(),
            new_train,
            new_val,
            new_test,
            evaluation,
        )
        params = model_config.get("learned_params", {})
        new_model.learn(0, params)
        new_model.reload()

        pred = new_model.forecast(torch.tensor(val_data.astype(np.float32)))
        preds.append(pred.numpy())

        pred_test = new_model.forecast(torch.tensor(test.astype(np.float32)))
        preds_test.append(pred_test.numpy())
        # print(f"{preds_test[-1].shape=}, {len(preds_test)=}, {preds_test[0:10]=}")
        #######################################################
        return (pred, pred_test)
