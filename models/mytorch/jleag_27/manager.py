import os
from enum import Enum
import importlib
import torch
import torchvision
import torch.nn as nn
from torch import optim
from enum import IntEnum
from .model import Model
from torch.amp import autocast
from contextlib import nullcontext
from sklearn.metrics import roc_auc_score
from sklearn.metrics import root_mean_squared_error


METRICS_LABEL1_NDX = 0
METRICS_PRED1_NDX = 1
METRICS_LOSS1_NDX = 2
METRICS_SIZE = 3

SAVED_LEARNING_DATA = "learning_dataset.npz"
SAVED_MODEL_NAME = "learned_model.pth"
SAVED_TMP_MODEL_NAME = "tmp_learned_model.pth"


class Diff(IntEnum):
    NEG = 0
    ZERO = 1
    POS = 2


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class BaseManager:
    def __init__(self, model, save_path, config):
        self.save_path = save_path
        self.config = config
        if os.path.isdir(save_path) is False:
            os.makedirs(save_path)

        self.model = model
        self.model.apply(init_weights)

        # 損失関数
        # self.criterion = nn.BCELoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.pos_weight = torch.tensor(self.config["pos_weight"], device=device)
        # self.criterion = nn.BCEWithLogitsLoss(
        # self.criterion_func = nn.BCEWithLogitsLoss
        self.criterion_func = nn.MSELoss
        self.criterion = nn.MSELoss()
        # self.criterion = nn.BCEWithLogitsLoss(
        #     pos_weight=self.pos_weight
        #     # pos_weight=1.0
        # )  # nn.BCELoss + sigmoid
        # self.criterion = nn.CrossEntropyLoss()
        self.save_path = save_path

    def set_loss_func(self, loss_params):
        is_optimize = self.config.get("is_optimize", False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if is_optimize or self.criterion is None:
            # loss_params = params["loss"]
            if "pos_weight" in loss_params:
                loss_params["pos_weight"] = torch.tensor(
                    loss_params["pos_weight"], device=device
                )
            self.criterion = self.criterion_func(**loss_params)  # nn.BCELoss + sigmoid

        # if is_optimize or self.optimizer is None:
        #     adam_params = params["adam"]
        #     self.optimizer = self.get_optimizer(adam_params)

    def get_model(self):
        return self.model

    def get_optimizer(self, sgd_params):
        # return optim.SGD(
        #     self.model.parameters(),
        #     lr=0.01,
        #     # betas=(0.9, 0.999),
        #     # eps=float(self.config["adam_eps"]),
        #     # lr=float(self.config["adam_lr"]),
        #     # betas=self.config["adam_betas"],
        #     # eps=1e-8,  # self.config["adam_eps"],
        # )

        # print(f"Using Adam optimizer with params: {params}")

        # beta1 = params["beta1"]
        # beta2 = params["beta2"]
        # params["beta"] = (params["beta1"], params["beta2"])
        # params.pop("beta1", None)
        # params.pop("beta2", None)
        print(f"SGD params: {sgd_params=}")
        print(f"SGD params: {self.model.parameters()=}")
        return optim.SGD(self.model.parameters(), **sgd_params)
        # return optim.Adam(
        #     self.model.parameters(),
        #     **adam_params,
        # )

    def get_tmp_model_name(self):
        return SAVED_TMP_MODEL_NAME

    def get_model_name(self):
        return SAVED_MODEL_NAME

    def get_path(self):
        path = os.path.join(os.path.dirname(__file__), "tmp")
        return path

    def get_mode(self):
        return self.mode

    def compute_batch_loss(
        self,
        model,
        batch_ndx,
        data_label,
        device,
        metrics,
        batch_max_size,
        is_optimize=False,
    ):
        (data, label) = data_label
        data = data.to(device)
        label = label.to(device)
        ctx = autocast(str(device)) if is_optimize else nullcontext()

        # is_optimizeがTrueのときはampを使用
        with ctx:
            prediction = model(data)
            loss = self.criterion(prediction, label)
            # print(loss.item())
            # print(f"{loss.item()=}")

        with torch.no_grad():
            start_ndx = batch_ndx * batch_max_size
            end_ndx = start_ndx + label.size(0)
            # tmp_prediction = prediction.detach()
            tmp_prediction = prediction
            # print(f"Train No.1 tmp_prediction={tmp_prediction[0:15]}")
            # print(f"Batch {batch_ndx}: {tmp_prediction=}, {label=}, {loss=}")
            metrics[METRICS_LABEL1_NDX, start_ndx:end_ndx] = label[:, 0].detach()
            metrics[METRICS_PRED1_NDX, start_ndx:end_ndx] = tmp_prediction[:, 0]
            metrics[METRICS_LOSS1_NDX, start_ndx:end_ndx] = (
                tmp_prediction[:, 0] - label[:, 0]
            )

            # loss_test = self.evaluate(metrics[:, start_ndx:end_ndx])
            # f1 = loss_test["f1"]
            # # print(f"Batch {batch_ndx}: {loss_test=}, {loss=}")
            # ration = 1.0
            # if f1 < 0.0001:
            #     ratio = 2.0
            # elif f1 < 0.1:
            #     ratio = 1.25
        return loss

    def evaluate(self, metrics_base):
        threshold_val = 0.0
        # print(f"{metrics_base[METRICS_LABEL1_NDX]=}")
        # print(f"{metrics_base[METRICS_PRED1_NDX]=}")
        # print(f"{metrics_base[METRICS_LOSS1_NDX]=}")
        # rmse = torch.sqrt()
        # auc = roc_auc_score(
        rmse = root_mean_squared_error(
            metrics_base[METRICS_LABEL1_NDX].cpu().numpy(),
            metrics_base[METRICS_PRED1_NDX].cpu().numpy(),
        )

        label_pos = metrics_base[:, metrics_base[METRICS_LABEL1_NDX] > threshold_val]
        label_neg = metrics_base[:, metrics_base[METRICS_LABEL1_NDX] <= threshold_val]
        pred_pos = metrics_base[:, metrics_base[METRICS_PRED1_NDX] > threshold_val]
        pred_neg = metrics_base[:, metrics_base[METRICS_PRED1_NDX] <= threshold_val]

        pos_true = metrics_base[
            :,
            (metrics_base[METRICS_LABEL1_NDX] > threshold_val)
            & (metrics_base[METRICS_PRED1_NDX] > threshold_val),
        ]
        pos_false = metrics_base[
            :,
            (metrics_base[METRICS_LABEL1_NDX] > threshold_val)
            & (metrics_base[METRICS_PRED1_NDX] <= threshold_val),
        ]
        neg_true = metrics_base[
            :,
            (metrics_base[METRICS_LABEL1_NDX] <= threshold_val)
            & (metrics_base[METRICS_PRED1_NDX] <= threshold_val),
        ]
        neg_false = metrics_base[
            :,
            (metrics_base[METRICS_LABEL1_NDX] <= threshold_val)
            & (metrics_base[METRICS_PRED1_NDX] > threshold_val),
        ]
        # print(f"{pos_true.shape=}, {pos_false.shape=}")
        # print(f"{neg_true.shape=}, {neg_false.shape=}")

        # print(f"{label_pos.shape=}, {label_neg.shape=}")
        # print(f"{pred_pos.shape=}, {pred_neg.shape=}")

        label_pos_count = label_pos.shape[1]
        label_neg_count = label_neg.shape[1]
        sample_cnt = label_pos_count + label_neg_count

        # true_pos = pred_pos[:, pred_pos[METRICS_PRED1_NDX] >= threshold_val]
        # neg_true = pred_neg[:, pred_neg[METRICS_PRED1_NDX] < threshold_val]

        pos_true_num = pos_true.shape[1]
        pos_false_num = label_pos_count - pos_true_num
        neg_true_num = neg_true.shape[1]
        neg_false_num = label_neg_count - neg_true_num
        # print(f"{label_pos_count=}, {label_neg_count=}")
        # print(f"{pos_true_num=}, {pos_false_num=}")
        # print(f"{neg_true_num=}, {neg_false_num=}")

        accuracy = (
            (pos_true_num + neg_true_num) / (sample_cnt) if sample_cnt != 0 else 0
        )
        recall = (
            pos_true_num / (pos_true_num + neg_false_num)
            if (pos_true_num + neg_false_num) != 0
            else 0
        )
        precision = (
            pos_true_num / (pos_true_num + pos_false_num)
            if (pos_true_num + pos_false_num) != 0
            else 0
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )
        # print(f"{accuracy=}, {f1=}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rmse": rmse,
            "ALL": 100 * f1,
        }

    def get_metrics_size(self):
        return METRICS_SIZE

    def log_metrics(self, epoch_ndx, mode_str, metrics, writer):
        results = self.evaluate(metrics)

        for str_key, value in results.items():
            writer.add_scalar(mode_str + ":" + str_key, value, epoch_ndx)
        return results["rmse"]
