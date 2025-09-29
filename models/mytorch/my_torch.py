import os
import importlib
import pandas as pd
import sys
from models.abstract_model import IModel

# from .bento_demand_24.model import Model
from .jleag_27.model import Model
from pathlib import Path
from torch.amp import GradScaler


# modelの切り替え
from .jleag_27.manager import BaseManager

import models.mytorch.jleag_27.model
import models.mytorch.jleag_27.manager

from torch.utils.tensorboard import SummaryWriter

import torch
from torch import optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from common.utility import Utility

import torch._dynamo


import time
import datetime


importlib.reload(models.mytorch.jleag_27.manager)

# BATCH_SIZE = 32
NUM_WORKERS = 4

torch._dynamo.config.suppress_errors = True


# 精度をわずかに下げて、計算速度を向上させる
torch.set_float32_matmul_precision("high")
# 進捗を表示する間隔(10なら10回学習ごとに一度経過を出力)
DISPLAY_STEP = 10


class MyTorch(IModel):
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
        self.start_epoch = 0
        if train is not None and val is not None:
            self.train_dataset = train
            # print(f"{len(train)=}, {train[0][0]=}")
            self.val_dataset = val
            input_size = self.train_dataset.get_column_number()
            output_size = self.train_dataset.get_label_number()
            # print(f"input_size={input_size}, output_size={output_size}")
            self.manager = BaseManager(
                Model(input_size, output_size, config["dropout"]),
                save_dir,
                config,
                # config["pos_weight"],
            )
            self.test_dataset = test
        self.continue_epoch = config["continue"]
        self.epoch = config["epoch"]
        self.config = config
        self.is_optimize = self.config.get("is_optimize", False)
        self.max_direction = True
        self.save_dir = save_dir
        if config.get("optimize_direction", False) == "minimize":
            self.max_direction = False
        self.load()

    @classmethod
    def get_name(self):
        current_dir = Path(__file__).resolve().parent.name
        return current_dir

    def get_board_log_path(self):
        return self.board_log_path

    def reload(self):
        print(f"reload model from {self.save_model_path}")
        checkpoint = torch.load(self.save_model_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = 0

    def load(self):
        tmp_model = self.manager.get_model()
        path = self.manager.get_path()
        saved_tmp_filename = self.manager.get_tmp_model_name()
        saved_filename = self.manager.get_model_name()

        self.board_log_path = os.path.join(self.save_dir, "board_log")
        self.save_tmp_model_path = os.path.join(self.save_dir, saved_tmp_filename)
        self.save_model_path = os.path.join(self.save_dir, saved_filename)
        self.writer = SummaryWriter(self.board_log_path)

        # 最適化のときだけpin_memoryを有効にする
        pin_memory = self.is_optimize

        self.train_batch = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )
        self.val_batch = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.compile(tmp_model.to(self.device))
        # self.optimizer = self.manager.get_optimizer()
        self.optimizer = None
        self.loss = 0
        self.best_score = 0
        self.start_epoch = 0

        if self.max_direction is False:
            self.best_score = 1e10

        if self.is_optimize:
            return

        if self.continue_epoch and os.path.isfile(self.save_tmp_model_path):
            print(f"{self.save_tmp_model_path=}")
            print(f"{self.save_dir=}")
            checkpoint = torch.load(self.save_tmp_model_path, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimu_params = self.get_optimu_params(None, self.config)
            self.optimizer = self.manager.get_optimizer(optimu_params)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"]
            self.loss = checkpoint["loss"]
            self.best_score = checkpoint["score"]

    def enumerate_with_estimate(self, iter, text, ndx, start_ndx=0):
        iter_len = len(iter)
        backoff = 2
        print_ndx = 5
        start_ndx = iter.num_workers

        start_ts = time.time()
        start_flag = True
        for batch_ndx, item in enumerate(iter):
            yield (batch_ndx, item)
            if batch_ndx == print_ndx:
                # 予想期間
                duration_sec = (
                    (time.time() - start_ts)
                    / (batch_ndx - start_ndx + 1)
                    * (iter_len - start_ndx)
                )
                total_sec = time.time() - start_ts
                done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)

                if batch_ndx == print_ndx and ndx % DISPLAY_STEP == 0:
                    if start_flag:
                        print(f"{text} starting-------------------------")
                        start_flag = False
                    print(
                        f"{text} {batch_ndx}/{iter_len} 経過時間:{(total_sec):.2f}秒, "
                        f"終了予定時間：{done_dt.strftime('%m/%d %H:%M')}, "
                        f"予測計測期間:{int(duration_sec/60)}分{(duration_sec%60):.2f}秒"
                    )
                    print_ndx *= backoff
                    if print_ndx == 40:
                        print_ndx = 50
                    elif print_ndx == 400:
                        print_ndx = 500

            if batch_ndx + 1 == start_ndx:
                start_ts = time.time()

            pass

    def do_training(self, ndx, epoch, batch, params, is_optimize):
        self.model.train()
        metrics_size = self.manager.get_metrics_size()
        trainMetrics = torch.zeros(metrics_size, len(batch.dataset), device=self.device)

        batch_iter = self.enumerate_with_estimate(
            iter=batch, text=f"E{ndx}/{epoch} Training:", ndx=ndx
        )
        # if is_optimize or self.optimizer is None:
        #     adam_params = params["adam"]
        #     self.optimizer = self.manager.get_optimizer(adam_params)

        scaler = GradScaler()
        for batch_ndx, data_label in batch_iter:
            self.optimizer.zero_grad()
            self.loss = self.manager.compute_batch_loss(
                self.model,
                batch_ndx,
                data_label,
                self.device,
                trainMetrics,
                batch.batch_size,
                is_optimize,
            )
            if is_optimize:
                self.loss.backward()
                self.optimizer.step()
            else:
                scaler.scale(self.loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        return trainMetrics.to("cpu")

    def do_validation(self, ndx, epoch, batch, is_optimize):
        self.model.eval()
        with torch.no_grad():
            metrics_size = self.manager.get_metrics_size()
            valMetrics = torch.zeros(
                metrics_size, len(batch.dataset), device=self.device
            )
            batch_iter = self.enumerate_with_estimate(
                iter=batch, text=f"E{ndx}/{epoch} Validation:", ndx=ndx
            )
            for batch_ndx, data_label in batch_iter:
                self.manager.compute_batch_loss(
                    self.model,
                    batch_ndx,
                    data_label,
                    self.device,
                    valMetrics,
                    batch.batch_size,
                    is_optimize,
                )
        return valMetrics.to("cpu")

    def forecast(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            prediction = self.model(data)

        # print(f"Foreact No.1 :tmp_prediction={prediction[0:15]}")
        return (prediction).to("cpu")[:, 0]

    def save_model(self, epoch_ndx, best_score, path):
        # path.find(".")
        # param_data_path = path.rsplit(".", 1)[0] + ".json"
        # info = {"params": float(some_numpy_value), "metrics": {"accuracy": 0.95}}
        print(f"save model to {path}: {epoch_ndx=}, {best_score=}")
        torch.save(
            {
                "epoch": epoch_ndx,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
                "score": best_score,
            },
            path,
        )

    def learn(self, i, params=None):
        # print(f"learn {i} start")
        ndx = 0
        start_at = time.time()
        # 最適化の際はepochを半分にする
        epoch = self.epoch
        if self.is_optimize:
            epoch = int(epoch / 2)

        self.manager.set_loss_func(params["loss"])
        self.optimizer = self.manager.get_optimizer(params["optim"])

        # valid_amp = self.config.get("amp", False)
        print(
            f"----------------------------Start {self.start_epoch}/{epoch}----------------------------"
        )
        for ndx in range(self.start_epoch, epoch):
            trnMetrics = self.do_training(
                ndx, self.epoch, self.train_batch, params, self.is_optimize
            )
            valMetrics = self.do_validation(
                ndx, self.epoch, self.val_batch, self.is_optimize
            )
            trn_score = self.manager.log_metrics(ndx, "trn", trnMetrics, self.writer)
            val_score = self.manager.log_metrics(ndx, "val", valMetrics, self.writer)
            score = (trn_score + val_score * 4) / 5.0

            if self.is_optimize:
                if ndx % 10 == 0:
                    print(
                        f"Trn:score={trn_score}, Val:score={val_score}, score={score}"
                    )
                # print(f"{self.max_direction=}, {self.best_score=}")
                if (
                    (score > self.best_score and self.max_direction and score > 0.5)
                    or (score <= self.best_score and not self.max_direction)
                ) and (ndx > self.epoch / 5):
                    self.best_score = score
                    self.save_model(ndx, self.best_score, self.save_model_path)
                continue

            if ndx % 10 == 0:
                print(
                    f"Trn:score={trn_score}, Val:score={val_score}, score={score}, Val:best_score={self.best_score}"
                )
                self.save_model(ndx, val_score, self.save_tmp_model_path)
            # print(f"{score=}, {self.best_score=}, {ndx=}, {self.epoch=}")
            if (
                (score > self.best_score and self.max_direction and (score > 0.50))
                or (score <= self.best_score and not self.max_direction)
            ) and (ndx > self.epoch / 5):
                self.best_score = score
                self.save_model(ndx, self.best_score, self.save_model_path)

        # todo 保存する条件を後で書くこと
        # self.save_model(ndx, self.best_score, self.save_model_path)
        print("----------------------------Evaluation----------------------------")
        self.writer.close()
        print("----------------------------End----------------------------")
        print(f" total time:{(time.time() - start_at):.2f}")
        print(f"execute:tensorboard --logdir {self.get_board_log_path()}")

    def get_optimu_params(self, trial, model_config):
        is_optimize = self.config.get("is_optimize", False)
        params = {}
        if is_optimize:
            optimu_config = self.config["optimize"].get("sgd", {})
            params = Utility().get_float_params(params, optimu_config, trial)
        else:
            tmp_params = model_config.get("learned_params", {})
            need_params = ["lr", "momentum", "weight_decay"]
            for name in tmp_params:
                if name in need_params:
                    params[name] = tmp_params[name]
        return params

    # def get_adam_params(self, trial, model_config):
    #     is_optimize = self.config.get("is_optimize", False)
    #     params = {}
    #     if is_optimize:
    #         adam_config = self.config["optimize"].get("adam", {})
    #         params = Utility().get_float_params(params, adam_config, trial)
    #         params = Utility().get_categorical_params(params, adam_config, trial)
    #     else:
    #         tmp_params = model_config.get("learned_params", {})
    #         need_params = ["beta1", "beta2", "eps", "lr", "weight_decay", "amsgrad"]
    #         for name in tmp_params:
    #             if name in need_params:
    #                 params[name] = tmp_params[name]
    #
    #     # print(f"adam_{params=}")
    #     params["betas"] = (params["beta1"], params["beta2"])
    #     params.pop("beta1", None)
    #     params.pop("beta2", None)
    #     # print(f"adam_{params=}")
    #     return params

    def get_loss_params(self, trial, model_config):
        is_optimize = self.config.get("is_optimize", False)
        params = {}
        if is_optimize:
            loss_config = self.config["optimize"].get("loss", {})
            # params = Utility().get_float_params(params, loss_config, trial)
            params = Utility().get_const_params(params, loss_config, trial)
        else:
            tmp_params = model_config.get("learned_params", {})
            need_params = ["pos_weight"]
            for name in tmp_params:
                if name in need_params:
                    params[name] = tmp_params[name]

        # print(f"loss1_{params=}")
        # print(f"loss2_{params=}")
        return params

    def get_torch_params(self, trial, model_config):
        is_optimize = self.config.get("is_optimize", False)
        params = {}
        if is_optimize:
            torch_config = self.config["optimize"].get("torch", {})
            params = Utility().get_categorical_params(params, torch_config, trial)
            # print(f"torch_{params=}")

            if params["scheduler"] == "StepLR":
                params["step_size"] = trial.suggest_int("step_size", 5, 30)
                params["gamma"] = trial.suggest_float("gamma", 0.1, 0.9)
            elif params["scheduler"] == "CosineAnnealingLR":
                params["t_max"] = trial.suggest_int("t_max", 10, 50)
        else:
            params = model_config.get("learned_params", {})
            # need_params = ["pos_weight", "weight"]
        return params

    def get_model_params(self, trial, model_config):
        is_optimize = self.config.get("is_optimize", False)
        # if is_optimize:
        params = {}
        optimu_params = self.get_optimu_params(trial, model_config)
        loss_params = self.get_loss_params(trial, model_config)
        torch_params = self.get_torch_params(trial, model_config)
        params["optim"] = optimu_params
        params["loss"] = loss_params
        params["torch"] = torch_params
        # adams_params
        # adams_config = model_config.get("adams", {})
        # adams_params[name] = trial.suggest_float(name, low, high, log=log_scale)
        return params

    def get_model_optimized_params(self, model_config):
        params = {}
        # adams_params = self.get_adam_params(None, model_config)
        optimu_params = self.get_optimu_params(None, model_config)
        loss_params = self.get_loss_params(None, model_config)
        torch_params = self.get_torch_params(None, model_config)
        params["optim"] = optimu_params
        params["loss"] = loss_params
        params["torch"] = torch_params
        return params

    def check_params(self, params):
        pass
