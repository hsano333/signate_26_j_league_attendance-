from models.abstract_model import IModel
from common.my_enum import MLTask
from pathlib import Path
from ensemble.oof import OOF
import numpy as np
import os


class Blending(IModel):
    DefaultDirectory = "blending_making_features"

    def __init__(self, config, task=MLTask.Classification, direction_maxmize=True):
        self.name = self.get_name()
        self.config = config
        # self.dataset_models = {}
        self.dir_path = Path(__file__).parent.resolve() / self.DefaultDirectory
        self.KFoldSplit = self.config["stacking"]["k_fold"]
        self.RANDAOM_STATE = self.config["stacking"]["random_state"]
        self.task = task
        self.oof = OOF(self.name, config, self.dir_path, task)
        self.direction_maxmize = direction_maxmize

    # def add_model(self, name, dataset, model, evaluation):
    #     # 最後はwegith
    #     self.dataset_models[name] = (model, dataset, evaluation, 1.0)

    def calc_score(self, scores):
        score = 0
        for name, (dataset, model, evaluation, _) in self.oof.dataset_models.items():
            score = score + self.oof.dataset_models[name][3] * scores[name]
        return score

    def calc_weight(self, scores):
        bottom_multi = 1
        bottom_add = 0

        # RMSEなど、小さいほど良いもの
        if self.direction_maxmize is False:
            for name, (_, _, _, _) in self.oof.dataset_models.items():
                bottom_multi = bottom_multi * scores[name]
            for name, (_, _, _, _) in self.oof.dataset_models.items():
                bottom_add += bottom_multi / scores[name]
            ratio = bottom_multi / bottom_add
            for name, (
                dataset,
                model,
                evaluation,
                weight,
            ) in self.oof.dataset_models.items():
                self.oof.dataset_models[name] = (
                    dataset,
                    model,
                    evaluation,
                    ratio / scores[name],
                )
        else:  # ausなど値が大きいほど良いもの
            for name, (_, _, _, _) in self.oof.dataset_models.items():
                bottom_add += scores[name]
            for name, (model, score, datase, _) in self.oof.dataset_models.items():
                self.oof.dataset_models[name] = (
                    dataset,
                    model,
                    evaluation,
                    ratio / scores[name],
                )

    # def blend(self):
    #     if len(images) != len(self.weights):
    #         raise ValueError("Number of images must match number of weights.")
    #
    #     blended_image = sum(w * img for w, img in zip(self.weights, images))
    #     return blended_image

    def get_name(self):
        return "Blending"
        # current_dir = Path(__file__).resolve().parent.name
        # return current_dir

    def reload(self):
        pass

    def learn(self, i, params=None):
        force_make = params.get("forece_make", False) if params else False
        self.oof.make_features(force_make)

        # combined_dataset = self.oof.make_features(True)
        # pred_data = combined_dataset.get_numpy_data()[0]
        # label = combined_dataset.get_numpy_data()[1]
        # test_data = combined_dataset.get_numpy_data()[2]

        self.calc_weight(self.oof.scores)
        score = self.calc_score(self.oof.scores)
        print(f"Blending Score: {score:.4f}")

        # ラベルはすべてのデータセットで同じなので、最初のデータセットから取得
        # (_, label, _) = next(iter(self.oof.dataset_models.values()))[0].get_numpy_data()
        # np.savez(file_cv_new_test_path, pred=result_pred, test=result_test)
        # print(f"Blending Model saved to {file_cv_new_test_path}")
        # return test_data

    def make_features(self, forece_make=False):
        return self.oof.make_features(forece_make)

    def add_model(self, name, dataset, model, evaluation):
        self.oof.add_model(name, dataset, model, evaluation)
        # self.dataset_models[name] = (dataset, model, evaluation)

    def clear_models(self):
        self.oof.dataset_models.clear()

    def forecast(self):
        combined_dataset = self.oof.make_features(False)
        self.calc_weight(self.oof.scores)
        score = self.calc_score(self.oof.scores)
        print(f"Blending Score: {score:.4f}")
        result_test = combined_dataset.get_numpy_data()[2]

        return result_test

    def get_model_params(self, trial, model_config):
        pass

    def get_model_optimized_params(self, model_config):
        pass

    def check_params(self, params):
        pass
