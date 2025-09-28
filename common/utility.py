import yaml
import shutil
from pathlib import Path
from common.my_enum import MLTask
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from dataclasses import dataclass

OriginalConfigFileName = "config_config.yaml"
ConfigFileName = "config.yaml"
OriginalConfigPath = (
    Path(__file__).parent.resolve() / OriginalConfigFileName
).as_posix()
ConfigPath = (Path(__file__).parent.resolve() / ConfigFileName).as_posix()
# config = Utility.load_yaml_config(config_path)


@dataclass
class Utility:
    def save_yaml_config(self, data):
        try:
            with open(ConfigPath, mode="w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True)
        except FileNotFoundError:
            print(f"{ConfigPath} not found. Using default settings.")
            raise FileNotFoundError()

    def load_yaml_config(self):
        # config_path = Path(__file__).parent.resolve() / "common" / ConfigFileName
        try:
            with open(ConfigPath, "r") as f:
                config = yaml.safe_load(f)
                return config
        except FileNotFoundError:
            print(f"{ConfigPath} not found. Using default settings.")
            raise FileNotFoundError()
        return None

    def delete_all_files(self, path):
        print(f"delete_all_files: {path=}")
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  # ファイルやシンボリックリンクを削除
            elif item.is_dir():
                shutil.rmtree(item)

    def make_kfold_directory(self, directory, kfold):
        if directory.exists():
            dir_count = len([f for f in directory.iterdir() if f.is_dir()])
            if dir_count != kfold:
                self.delete_all_files(directory)
                print("delete file in " + directory.as_posix())
        else:
            directory.mkdir(parents=True, exist_ok=True)

    def get_kf(self, task, sp_num, random_state):
        if task == MLTask.Regression:
            # 回帰にはKFoldを使う
            kf = KFold(n_splits=sp_num, shuffle=True, random_state=random_state)
        elif task == MLTask.Classification:
            # 分類にはStratifiedKFoldを使う
            kf = StratifiedKFold(
                n_splits=sp_num, shuffle=True, random_state=random_state
            )
        elif task == MLTask.TimeSeries:
            # 概念ドリフト(時系列分析)がある場合はTimeSeriesSplitを使う
            kf = TimeSeriesSplit(n_splits=sp_num)
        else:
            # 未定義
            kf = StratifiedKFold(
                n_splits=sp_num, shuffle=True, random_state=random_state
            )
        return kf

    def update_const_params(self, config):
        if "const" in config["optimize"] and config["optimize"]["const"] is not None:
            for name, bounds in config["optimize"].get("const", {}).items():
                # low, high = bounds
                config["learned_params"][name] = bounds

    def get_const_params(self, params, config_opt, trial, except_param=[]):
        if config_opt.get("const") is None:
            return params
        for name, bounds in config_opt.get("const", {}).items():
            if name in except_param:
                continue
            params[name] = bounds
        return params

    def get_int_params(self, params, config_opt, trial, except_param=[]):
        if config_opt.get("int") is None:
            return params
        for name, bounds in config_opt.get("int", {}).items():
            if name in except_param:
                continue
            low, high = bounds
            params[name] = trial.suggest_int(name, low, high)
        return params

    def get_categorical_params(self, params, config_opt, trial, except_param=[]):
        if config_opt.get("categorical") is None:
            return params
        for name, choices in config_opt.get("categorical", {}).items():
            if name in except_param:
                continue
            tmp = trial.suggest_categorical(name, choices)
            if tmp == "none":
                params[name] = None
            else:
                params[name] = tmp
        return params

    def get_float_params(self, params, config_opt, trial, except_param=[]):
        if config_opt.get("float") is None:
            return params
        for name, bounds in config_opt.get("float", {}).items():
            if name in except_param:
                continue
            if len(bounds) == 3:  # [low, high, log]
                low, high, log_scale = bounds
                params[name] = trial.suggest_float(
                    name, float(low), float(high), log=log_scale
                )
            else:
                low, high = bounds
                # print(f"{name=} {low=} {high=}, {trial=}")
                params[name] = trial.suggest_float(name, float(low), float(high))
        return params

    def get_model_params(self, trial, model_config):
        params = {}
        config_opt = model_config["optimize"]
        params = Utility().get_categorical_params(params, config_opt, trial)
        params = Utility().get_int_params(params, config_opt, trial)
        params = Utility().get_float_params(params, config_opt, trial)
        params = Utility().get_const_params(params, config_opt, trial)
        return params
