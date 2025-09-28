from common.utility import Utility
from dataset.src.simple_dataset import SimpleDataset
import torch
import numpy as np
import os
import yaml
import optuna


class OptunaOptimizer:
    def __init__(self, model_info, eval, evaluation):
        self.model = model_info.model
        self.model_name = model_info.model_name
        self.dataset = model_info.dataset
        self.task = model_info.task
        print(f"Task: {self.task}")
        print(f"Task: {self.task}")
        print(f"Task: {self.task}")
        print(f"Task: {self.task}")
        print(f"Task: {self.task}")
        print(f"Task: {self.task}")
        print(f"Task: {self.task}")
        print(f"Task: {self.task}")
        print(f"Task: {self.task}")
        self.directory = model_info.directory
        self.transform_label = model_info.transform_label
        self.evaluate = model_info.evaluate
        self.KFoldSplit = model_info.config["stacking"]["k_fold"]
        self.RANDAOM_STATE = model_info.config["stacking"]["random_state"]
        # self.config = model_info.config[self.model_name]
        self.config = model_info.config
        self.model_config = self.config["model"][self.model_name]
        self.optimize_direction = self.model_config["optimize_direction"]

    def calc_eval(self, i, preds, label):
        return self.evaluate(label, preds) * (i + 1) / self.KFoldSplit

    def __call__(self, trial):
        kf = Utility().get_kf(self.task, self.KFoldSplit, self.RANDAOM_STATE)
        (data, label, test) = self.dataset.get_numpy_data()
        label_number = self.dataset.get_label_number()
        tmp_name = self.dataset.get_name()

        if label.ndim == 1:
            split_label = label
        else:
            split_label = label[:, 0]

        y_prob_all = np.zeros(len(data))
        y_true_all = np.zeros(len(data))
        for i, (tr_idx, val_idx) in enumerate(kf.split(data, split_label)):
            k_directory = self.directory / f"fold_{i + 1}"
            train_data, val_data = data[tr_idx], data[val_idx]
            train_label, val_label = label[tr_idx], label[val_idx]
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

            new_model = self.model(
                self.model_config,
                k_directory.as_posix(),
                new_train,
                new_val,
                new_test,
                self.evaluate,
                trial,
            )
            params = new_model.get_model_params(trial, self.model_config)
            new_model.check_params(params)
            new_model.learn(i, params)
            new_model.reload()
            pred = new_model.forecast(torch.tensor(val_data.astype(np.float32)))
            y_prob_all[val_idx] = pred.numpy().reshape(-1)  # Ensure y_prob_all is 1D
            # y_true_all[val_idx] = val_label[:, 0]
            y_true_all[val_idx] = val_label.reshape(-1)  # Ensure y_true_all is 1D

            eval = self.calc_eval(i, y_prob_all, y_true_all)
            trial.report(eval, step=i)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        eval = self.calc_eval(self.KFoldSplit - 1, y_prob_all, y_true_all)
        print(f"Evaluation: {eval}")
        return eval
