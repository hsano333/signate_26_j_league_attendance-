from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score
import shutil
from common.utility import Utility
from common.model_info import ModelInfo
from eval.binary_analysis import BinaryAnalysis
from optimizer.optuna_optimizer import OptunaOptimizer

from pathlib import Path

from dataset.src.combined_dataset import CombinedDataset
from dataset.src.simple_dataset import SimpleDataset
from common.my_enum import MLTask
import torch
import numpy as np
import os
import yaml
import optuna


class MyOptuna:
    DefaultDirectory = "optimization"

    def __init__(self, config, task):
        self.dataset_models = {}
        self.config = Utility().load_yaml_config()
        self.task = task
        self.config = config
        self.dir_path = Path(__file__).parent.resolve() / self.DefaultDirectory
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.KFoldSplit = self.config["stacking"]["k_fold"]

    # def optimize(self):
    #     if len(self.dataset_models) != 1:
    #         raise ValueError(
    #             "Optimizer requires exactly one dataset and one model to learn."
    #         )

    def add_model(self, name, dataset, model, evaluation):
        self.dataset_models[name] = (dataset, model, evaluation)

    def clear_models(self):
        self.dataset_models.clear()

    # def optimize(self, model, model_name, dataset, task, directory, config):
    def optimize(self, model_info, eval, evaluation):
        opt = OptunaOptimizer(model_info, eval, evaluation)
        # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        # デフォルトのまま
        pruner = optuna.pruners.MedianPruner()
        # print(f"{opt.model_config=}")
        study = optuna.create_study(
            direction=opt.optimize_direction,
            pruner=pruner,
        )
        opt.model_config["is_optimize"] = True
        study.optimize(opt, n_trials=self.config["optimization"]["n_trials"])
        opt.model_config["is_optimize"] = False
        # print(f"{study.best_params=}")
        # print(f"{opt.config=}")
        opt.model_config["learned_params"] = study.best_params
        # print(f"pre:{opt.model_config["learned_params"]=}")
        Utility().update_const_params(opt.model_config)
        # print(f"post:{opt.model_config["learned_params"]=}")
        # opt.model_config["learned_params"] = study.best_params
        # opt.model_config.learn_params.update(
        #     {k: v for k, v in study.best_params.items()}
        # )
        # print(f"update:{opt.model_config=}")
        return opt.model_config

    def optimize_models(self, task, forece_make=False):
        preds = []
        tests = []

        for name, (dataset, model, evaluation) in self.dataset_models.items():
            model_name = name
            model_config = self.config["model"][model_name]
            dataset_name = dataset.get_name()
            forece_make = forece_make or model_config["optimize_overwrite"]

            name = dataset_name + "_" + model_name + "_k" + str(self.KFoldSplit)
            directory = self.dir_path / name
            optimized_config_path = (directory / "optimized_param.yaml").as_posix()
            Utility().make_kfold_directory(directory, self.KFoldSplit)
            # eval = BinaryAnalysis().accuracy
            # eval = evaluation
            trnsform_label = BinaryAnalysis().transform_label

            if not os.path.isfile(optimized_config_path) or forece_make:
                model_info = ModelInfo(
                    model,
                    model_name,
                    dataset,
                    task,
                    directory,
                    evaluation,
                    trnsform_label,
                    self.config,
                )
                # directory.mkdir(parents=True, exist_ok=True)
                # (pred, test) = self.make_new_feature(
                #     model, model_name, dataset, task, directory
                # )
                optimized_data = self.optimize(model_info, eval, evaluation)
                with open(optimized_config_path, mode="w", encoding="utf-8") as f:
                    yaml.safe_dump(optimized_data, f)
                self.config["model"][model_name] = optimized_data
                print("make optimized data:" + optimized_config_path)
            else:
                with open(optimized_config_path, "r") as f:
                    self.config["model"][model_name] = yaml.safe_load(f)
        # print(f"{self.config=}")

        # 10未満はテストなので保存しない
        if self.config["optimization"]["n_trials"] >= 10:
            Utility().save_yaml_config(self.config)
        else:
            print("test mode, not save config")
