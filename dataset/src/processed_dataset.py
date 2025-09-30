# from enum import auto
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from dataset.src.simple_dataset import SimpleDataset
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import Normalizer


from common.my_enum import MLTask

# from models.linear_regression.my_linear_regression import MyLinearRegression
from dataset.src.time_dataset import TimeDataset

import joblib

import torch
import numpy as np
import pandas as pd
import os


class ProcessedDataset(Dataset):
    def __init__(self, config, name="processed_dataset"):
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        self.TRAIN_PATH = os.path.join(current_dir, "data", "original", "train.csv")
        self.TEST_PATH = os.path.join(current_dir, "data", "original", "test.csv")
        self.TRAIN_ADD_PATH = os.path.join(
            current_dir, "data", "original", "train_add.csv"
        )

        self.TEST_ADD_PATH = os.path.join(
            current_dir, "data", "original", "train_add.csv"
        )

        self.STADIUM_PATH = os.path.join(current_dir, "data", "original", "stadium.csv")

        self.CONDITION_PATH = os.path.join(
            current_dir, "data", "original", "condition.csv"
        )
        self.CONDITION_ADD_PATH = os.path.join(
            current_dir, "data", "original", "condition_add.csv"
        )

        self.PREPOCESSED_PATH = os.path.join(
            current_dir, "data", "original", "dataset_preprocessed.pkl"
        )

        self.dateset_name = name
        self.config = config
        self.target_feature = self.config["target_feature"]

    def get_numpy_data(self):
        return (self.data.numpy(), self.label.numpy(), self.test_data.numpy())

    def get_test_id(self):
        return self.test_id

    def set_raw_label(self, raw_label):
        # pandas
        self.raw_label = raw_label

    def get_raw_label(self):
        # pandas
        return self.raw_label

    def convert_np_to_torch(self, data):
        return torch.tensor(data.values.astype(np.float32))

    def get_label_scaler(self):
        return self.label_scaler

    def get_mean_df(self):
        return (self.train_mean, self.test_mean)

    def convert_onehot_encoder(self, encoder, data, name):
        encoderd_data = encoder.transform(data[[name]])
        feature_names = encoder.get_feature_names_out([name])
        tmp_df = pd.DataFrame(encoderd_data, columns=feature_names)
        return pd.concat([data, tmp_df], axis=1)

    def convert_oneshot(self, data, name):
        encoded_data = pd.get_dummies(
            data,
            columns=[
                name,
            ],
        )
        return encoded_data

    def transform_label(self, task):
        if task == MLTask.Regression:
            self.label_number = self.label.shape[0]
            # print(f"{type(self.label)=}")
            # self.label = torch.from_numpy(self.label)
        else:
            unique = self.raw_label.unique()
            self.le = LabelEncoder()
            raw_np_label = self.le.fit_transform(self.label)
            self.label = torch.from_numpy(raw_np_label)
            self.label_number = unique.shape[0]

    def make_date_id(self):
        start_date = "2013-11-18"
        end_date = "2014-12-01"

        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame(all_dates, columns=["date"])
        df["datetime"] = pd.to_datetime(df["date"])
        df["weekday"] = df["datetime"].dt.day_name()
        # df["weekday_i"] = df["date"].dt.dayofweek

        df = df.reset_index()
        df = df.rename(columns={"index": "id"})
        df["week_id"] = df["id"] // 7
        # print(f"{df["week_id"]=}")
        df = df.drop(columns=["id", "date"])

        return df

    def get_mean_y(self, data, onehot_encoder):
        def trimmed_mean(x):
            if len(x) >= 3:
                return x.sort_values().iloc[0:-2].mean()
            else:
                return x.mean()  # データ数が2以下なら普通に平均

        mean_y = data.groupby("week_id")["y"].agg(trimmed_mean).reset_index()
        tmp_label = mean_y["y"]

        # data["y_roll"] = data["y"].rolling(window=10, min_periods=1).mean()
        # mean_y = data.groupby("week_id")["y_roll"].agg(trimmed_mean).reset_index()
        # tmp_label = mean_y["y_roll"]
        # data = data.dropna(subset=["y_roll"])

        tmp_data = mean_y["week_id"]

        liner = LinearRegression()
        # print(f"{tmp_data.shape=}, {tmp_label.shape=}")
        self.model = liner.fit(np.array(tmp_data).reshape(-1, 1), tmp_label)

        return self.model

    def adjust_rank(self, data):
        data["first1"] = False
        data["first2"] = False
        data["second"] = False
        data["second_last"] = False
        data["last"] = False
        data.loc[(data["section"] == 1), "first1"] = True
        data.loc[(data["section"] == 2), "first2"] = True
        # data.loc[(data["section"] >= 1) & (data["section"] <= 2), "first"] = True
        data.loc[(data["second"] >= 3) & (data["section"] <= 7), "second"] = True
        data.loc[(data["stage"] == "J1") & (data["section"] >= 29), "second_last"] = (
            True
        )
        data.loc[(data["stage"] == "J2") & (data["section"] >= 36), "second_last"] = (
            True
        )
        data.loc[(data["stage"] == "J1") & (data["section"] >= 33), "last"] = True
        data.loc[(data["stage"] == "J2") & (data["section"] >= 41), "last"] = True

        data["home_top_three"] = False
        data["home_bottom_three"] = False
        data["home_second_top_three"] = False
        data["home_second_bottom_three"] = False
        data["away_top_three"] = False
        data["away_bottom_three"] = False
        data["away_second_top_three"] = False
        data["away_second_bottom_three"] = False

        data.loc[(data["home_rank"] <= 3), "home_top_three"] = True
        data.loc[
            (data["home_rank"] <= 6) & (data["home_rank"] >= 4),
            "home_second_top_three",
        ] = True

        data.loc[(data["away_rank"] <= 3), "away_top_three"] = True
        data.loc[
            (data["away_rank"] <= 6) & (data["away_rank"] >= 4),
            "away_second_top_three",
        ] = True

        data.loc[
            (data["stage"] == "J1") & (data["home_rank"] >= 15), "home_bottom_three"
        ] = True
        data.loc[
            (data["stage"] == "J1")
            & (data["home_rank"] >= 12)
            & (data["home_rank"] <= 14),
            "home_second_bottom_three",
        ] = True

        data.loc[
            (data["stage"] == "J1") & (data["away_rank"] >= 15), "away_bottom_three"
        ] = True
        data.loc[
            (data["stage"] == "J1")
            & (data["away_rank"] >= 12)
            & (data["away_rank"] <= 14),
            "away_second_bottom_three",
        ] = True

        data.loc[
            (data["stage"] == "J2") & (data["home_rank"] >= 19), "home_bottom_three"
        ] = True
        data.loc[
            (data["stage"] == "J2")
            & (data["home_rank"] >= 16)
            & (data["home_rank"] <= 18),
            "home_second_bottom_three",
        ] = True

        data.loc[
            (data["stage"] == "J2") & (data["away_rank"] >= 19), "away_bottom_three"
        ] = True
        data.loc[
            (data["stage"] == "J2")
            & (data["away_rank"] >= 16)
            & (data["away_rank"] <= 18),
            "away_second_bottom_three",
        ] = True

        # data.loc[(data["stage"] == "J2") & (data["section"] >= 41), "last"] = True

        data["home_second_last_top"] = False
        data["away_second_last_top"] = False
        data["home_second_last_bottom"] = False
        data["away_second_last_bottom"] = False
        data["home_last_top"] = False
        data["away_last_top"] = False
        data["home_last_bottom"] = False
        data["away_last_bottom"] = False
        data["home_last_second_top"] = False
        data["away_last_second_top"] = False
        data["home_last_second_bottom"] = False
        data["away_last_second_bottom"] = False

        data.loc[
            (data["last"]) & (data["home_top_three"]),
            "home_last_top",
        ] = True
        data.loc[
            (data["last"]) & (data["away_top_three"]),
            "away_last_top",
        ] = True
        data.loc[
            (data["last"]) & (data["home_bottom_three"]),
            "home_last_bottom",
        ] = True
        data.loc[
            (data["last"]) & (data["away_bottom_three"]),
            "away_last_bottom",
        ] = True

        data.loc[
            (data["last"]) & (data["home_second_top_three"]),
            "home_last_second_top",
        ] = True
        data.loc[
            (data["last"]) & (data["away_second_top_three"]),
            "away_last_second_top",
        ] = True
        data.loc[
            (data["last"]) & (data["home_second_bottom_three"]),
            "home_last_second_bottom",
        ] = True
        data.loc[
            (data["last"]) & (data["away_second_bottom_three"]),
            "away_last_second_bottom",
        ] = True

        return data

    def mypreprocessing1(self, data):
        ##### data['day'] = data['gameday'].apply(lambda x: x[:4]).astype(int)
        data["month"] = data["gameday"].apply(lambda x: x[:2]).astype(int)
        data["weekday"] = data["gameday"].apply(lambda x: x[6])

        data["holiday"] = False
        data.loc[
            (data["gameday"].str.contains("祝")) | (data["gameday"].str.contains("休")),
            "holiday",
        ] = True

        # timeから時間を取り出す
        data["hour"] = data["time"].apply(lambda x: x.split(":")[0]).astype(int)

        bins = [0, 13, 17, 18, 24]
        labels = ["hour1", "hour2", "hour3", "hour4"]
        data["hour"] = pd.cut(data["hour"], bins=bins, labels=labels, right=True)

        # tvからサービスの数をカウント
        data["num_tv"] = data["tv"].apply(lambda x: len(x.split("／")))

        data["nhk1"] = False
        data["nhk2"] = False
        data["nhk_bs"] = False
        data.loc[(data["tv"].str.contains("ＮＨＫ")), "nhk1"] = True
        data.loc[(data["tv"].str.contains("ＮＨＫ総合")), "nhk2"] = True
        data.loc[(data["tv"].str.contains("ＮＨＫ　ＢＳ")), "nhk_bs"] = True
        # data.loc[(data["tv"].str.contains("ＫＢＳ京都")), "kbs_kyoto"] = True

        data["other_tv"] = False
        data.loc[
            (data["tv"].str.contains("テレ"))
            | (data["tv"].str.contains("ＴＯＫＹＯ"))
            | (data["tv"].str.contains("放送"))
            | (data["tv"].str.contains("ＫＢＳ京都")),
            "other_tv",
        ] = True

        data["rain"] = False
        data["snow"] = False
        data.loc[
            data["weather"].str.contains("雨"),
            "rain",
        ] = True
        data.loc[
            (data["weather"].str.contains("雪")),
            "snow",
        ] = True

        data = self.adjust_rank(data)

        data["humidity"] = data["humidity"].str.replace("%", "").astype(float)

        # data["home_score"] = data["home_score"] / data["section"]
        # data["away_score"] = data["away_score"] / data["section"]
        data = data.drop(
            [
                # "rank_sum",
                "id",
                "gameday",
                "time",
                "stadium",
                "tv",
                "weather",
                # "temperature",
                "humidity",
                "home_score",
                "away_score",
                "month",
                "match",
                "section",
                # "mean",
                # "home",
                "count",
                # "num_tv",
                "std",
                # "capa",
                "t25",
                "t75",
                "t50",
                "min",
                "max",
                # "home_rank",
                # "away_rank",
                "mean",
                "home_top_three",
                "home_bottom_three",
                "home_second_top_three",
                "home_second_bottom_three",
                "away_top_three",
                "away_bottom_three",
                "away_second_top_three",
                "away_second_bottom_three",
                "home_second_last_top",
                "away_second_last_top",
                "home_second_last_bottom",
                "away_second_last_bottom",
                "home_last_top",
                "away_last_top",
                "home_last_bottom",
                "away_last_bottom",
                "home_last_second_top",
                "away_last_second_top",
                "home_last_second_bottom",
                "away_last_second_bottom",
            ],
            axis=1,
        )
        # data = data.drop(
        #     [
        #         "id",
        #         "match",
        #         "gameday",
        #         "time",
        #         "stadium",
        #         "tv",
        #         "weather",
        #         "temperature",
        #         "humidity",
        #         "home_score",
        #         "away_score",
        #         "month",
        #     ],
        #     axis=1,
        # )
        return data

    def mypreprocessing2(self, data, onehot_encoder):
        return data.astype(float)

    def cal_score(self, row):
        if row["home_score"] == row["away_score"]:
            row["sum_score_" + row["home"]] = 1 + row["home_score"] * 0.001
            row["sum_score_" + row["away"]] = 1 + row["away_score"] * 0.001
        elif row["home_score"] > row["away_score"]:
            row["sum_score_" + row["home"]] = 3 + row["home_score"] * 0.001
            row["sum_score_" + row["away"]] = 0 + row["away_score"] * 0.001
        else:
            row["sum_score_" + row["away"]] = 3 + row["away_score"] * 0.001
            row["sum_score_" + row["home"]] = 0 + row["home_score"] * 0.001
        return row

    def calc_win_rate(self, melted, names):
        melted["win_rate"] = 0
        melted = melted.sort_values(["year", "section"])
        for name in names:
            melted.loc[melted["team"] == name, "win_rate"] = melted["sum_score_" + name]
            melted.loc[melted["team"] == name, "win_rate"] = melted.loc[
                melted["team"] == name
            ]["win_rate"].shift(1)
            melted.loc[melted["win_rate"].isnull(), "win_rate"] = 100
        return melted

    def calc_rank(self, df):
        all_teams_name = df["home"].unique().tolist()
        all_teams_sum_score = ["sum_score_" + x for x in all_teams_name]
        scores = all_teams_sum_score
        tmp_columns = df.columns.tolist() + scores

        df_scores = df.apply(self.cal_score, axis=1)
        df_scores = df_scores.fillna(0)
        df_scores = df_scores.sort_values(["year", "section"])

        df_scores_2012 = df_scores[df_scores["year"] == 2012].copy()
        df_scores_2013 = df_scores[df_scores["year"] == 2013].copy()
        df_scores_2014 = df_scores[df_scores["year"] == 2014].copy()

        df_scores_2012[scores] = df_scores_2012[scores].cumsum()
        df_scores_2013[scores] = df_scores_2013[scores].cumsum()
        df_scores_2014[scores] = df_scores_2014[scores].cumsum()

        df_scores = pd.concat([df_scores_2012, df_scores_2013, df_scores_2014])
        df_scores = df_scores.sort_values(["year", "section"])

        df_scores_J1 = df_scores[df_scores["stage"] == "J1"]
        df_scores_J2 = df_scores[df_scores["stage"] == "J2"]

        df_scores_J1 = df_scores_J1.sort_values(["year", "section"])
        df_scores_J2 = df_scores_J2.sort_values(["year", "section"])

        names_J1_base = df_scores_J1["home"].dropna().unique().tolist()
        names_J2_base = df_scores_J2["home"].dropna().unique().tolist()
        names_J1 = ["sum_score_" + x for x in names_J1_base]
        names_J2 = ["sum_score_" + x for x in names_J2_base]

        rank_J1_df = df_scores_J1[names_J1].rank(axis=1, ascending=False, method="min")
        rank_J2_df = df_scores_J2[names_J2].rank(axis=1, ascending=False, method="min")

        df_scores_J1 = df_scores_J1.drop(names_J1, axis=1)
        df_scores_J2 = df_scores_J2.drop(names_J2, axis=1)

        con_rank_J1 = pd.concat([df_scores_J1, rank_J1_df], axis=1)
        con_rank_J2 = pd.concat([df_scores_J2, rank_J2_df], axis=1)

        teams_match = ["year", "section"] + scores

        melted_J1 = con_rank_J1.melt(
            value_vars=["home", "away", "stage"],
            value_name="team",
            id_vars=names_J1 + ["year", "section", "stage"],
        )
        melted_J2 = con_rank_J2.melt(
            value_vars=["home", "away", "stage"],
            value_name="team",
            id_vars=names_J2 + ["year", "section", "stage"],
        )

        melted_J1 = self.calc_win_rate(melted_J1, names_J1_base)
        melted_J2 = self.calc_win_rate(melted_J2, names_J2_base)

        standing_J1 = melted_J1[["year", "section", "stage", "team", "win_rate"]]
        standing_J2 = melted_J2[["year", "section", "stage", "team", "win_rate"]]
        standing_df = pd.concat([standing_J1, standing_J2])
        # standing_df['win_rate2'] = standing_df['win_rate']

        df["home_rank"] = 0
        df["away_rank"] = 0

        pd.merge(
            df,
            standing_df,
            left_on=["year", "section", "stage", "away"],
            right_on=["year", "section", "stage", "team"],
        )
        df_home = pd.merge(
            df,
            standing_df,
            left_on=["year", "section", "stage", "home"],
            right_on=["year", "section", "stage", "team"],
            how="left",
        )
        df_away = pd.merge(
            df,
            standing_df,
            left_on=["year", "section", "stage", "away"],
            right_on=["year", "section", "stage", "team"],
            how="left",
        )
        df["home_rank"] = df_home["win_rate"]
        df["away_rank"] = df_away["win_rate"]
        df["rank_sum"] = df["home_rank"] + df["away_rank"]

        #    df.loc[df['stage'] == 'J1', "home_rank_section"] = df['home_rank'] / df['section'] * 34
        # df.loc[df['stage'] == 'J1', "away_rank_section"] = df['away_rank'] / df['section'] * 34
        # df.loc[df['stage'] == 'J2', "home_rank_section"] = df['home_rank'] / df['section'] * 42
        # df.loc[df['stage'] == 'J2', "away_rank_section"] = df['away_rank'] / df['section'] * 42
        return df

    def load(self):
        if os.path.exists(self.PREPOCESSED_PATH) and self.config["overwrite"] is False:
            print(f"Loading preprocessed dataset from {self.PREPOCESSED_PATH}")
            loaded_data = joblib.load(self.PREPOCESSED_PATH)
            self.data = loaded_data["data"]
            self.label = loaded_data["label"]
            self.label_number = self.label.shape[1]
            self.test_data = loaded_data["test_data"]
            self.test_id = loaded_data["datetime"]
            self.raw_label = loaded_data["raw_label"]
            self.target_feature = loaded_data["target_feature"]
            return

        train = pd.read_csv(self.TRAIN_PATH)
        train_add = pd.read_csv(self.TRAIN_ADD_PATH)
        test = pd.read_csv(self.TEST_PATH)
        test_add = pd.read_csv(self.TEST_ADD_PATH)
        stadium = pd.read_csv(self.STADIUM_PATH)
        condition = pd.read_csv(self.CONDITION_PATH)
        condition_add = pd.read_csv(self.CONDITION_ADD_PATH)

        full_train = pd.concat([train, train_add], axis=0)
        full_condition = pd.concat([condition, condition_add], axis=0)

        stadium = stadium.rename(columns={"name": "stadium"})
        stadium = stadium[["stadium", "capa"]]
        full_condition = full_condition[
            ["id", "home_score", "away_score", "weather", "temperature", "humidity"]
        ]

        full_train = pd.merge(full_train, stadium, on="stadium", how="left")
        full_train = pd.merge(full_train, full_condition, on="id", how="left")
        full_test = pd.merge(test, stadium, on="stadium", how="left")
        full_test = pd.merge(full_test, full_condition, on="id", how="left")

        # full_train = full_train.iloc[100:]

        self.test_id = full_test["id"]
        self.raw_label = full_train[self.target_feature]

        full_train.loc[full_train["stage"] == "Ｊ１", "stage"] = "J1"
        full_train.loc[full_train["stage"] == "Ｊ２", "stage"] = "J2"

        full_test.loc[full_test["stage"] == "Ｊ１", "stage"] = "J1"
        full_test.loc[full_test["stage"] == "Ｊ２", "stage"] = "J2"

        full_train["section"] = (
            full_train["match"].apply(lambda x: x.split("節")[0][1:]).astype(int)
        )
        full_test["section"] = (
            full_test["match"].apply(lambda x: x.split("節")[0][1:]).astype(int)
        )

        stadium = stadium.rename(columns={"name": "stadium"})
        stadium = stadium[["stadium", "capa"]]

        full_train = self.calc_rank(full_train)
        full_test = self.calc_rank(full_test)

        full_train2_train = full_train[
            ((full_train["stage"] == "J1") & (full_train["section"] <= 17))
            | ((full_train["stage"] == "J2") & (full_train["section"] <= 24))
        ]

        # full_train = full_train[
        #     ((full_train["stage"] == "J1") & (full_train["section"] > 5))
        #     | ((full_train["stage"] == "J2") & (full_train["section"] > 5))
        # ]

        # full_train = full_train2_train.copy()
        # self.raw_label = full_train[self.target_feature]

        full_train_2012_train = full_train2_train[full_train2_train["year"] == 2012]
        full_train_2013_train = full_train2_train[full_train2_train["year"] == 2013]
        full_train_2014_train = full_train2_train[full_train2_train["year"] == 2014]
        full_train2_train_2012_grouped = full_train_2012_train.groupby("home")["y"]
        full_train2_train_2013_grouped = full_train_2013_train.groupby("home")["y"]
        full_train2_train_2014_grouped = full_train_2014_train.groupby("home")["y"]

        describe = ["count", "mean", "std", "min", "t25", "t50", "t75", "max"]

        # full_train2_train[['count' , 'mean', 'std', 'min', 't25', 't50', 't75', 'max']] = full_train2_train['home'].apply(lambda x: grouped.get_group(x).describe())

        # full_train["mean"] = 1.0
        full_train.loc[full_train["year"] == 2012, describe] = full_train[
            full_train["year"] == 2012
        ]["home"].apply(
            lambda x: full_train2_train_2012_grouped.get_group(x).describe()
        )

        full_train.loc[full_train["year"] == 2013, describe] = full_train[
            full_train["year"] == 2013
        ]["home"].apply(
            lambda x: full_train2_train_2013_grouped.get_group(x).describe()
        )

        full_train.loc[full_train["year"] == 2014, describe] = full_train[
            full_train["year"] == 2014
        ]["home"].apply(
            lambda x: full_train2_train_2014_grouped.get_group(x).describe()
        )

        # self.raw_label = full_train2_train[self.target_feature]
        # full_train = full_train2_train

        full_test[describe] = full_test["home"].apply(
            lambda x: full_train2_train_2014_grouped.get_group(x).describe()
        )
        # full_train["y"] = full_train["y"] - full_train["mean"]

        # print(f"No.1:{full_train.shape=}")
        # full_train = pd.concat(
        #     [full_train_2012_train, full_train_2013_train, full_train_2014_train],
        #     axis=0,
        # )
        # print(f"No.2:{full_train.shape=}")

        self.train_mean = full_train["mean"]
        self.test_mean = full_test["mean"]

        full_train = self.mypreprocessing1(full_train)
        full_test = self.mypreprocessing1(full_test)

        full_train = full_train.drop([self.target_feature], axis=1)

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )

        # drop_col = ["capa", "home_rank", "away_rank"]
        numerical_features = [
            "capa",
            "home_rank",
            "away_rank",
            "rank_sum",
            # "mean",
            # "home_score",
            # "away_score",
            # "max",
            # "min",
            # "t25",
            # "t50",
            # "t75",
            # "home_score",
            # "away_score",
            # "std",
            "temperature",
            # "humidity",
        ]
        # numerical_features = ["home_rank", "away_rank"]
        # kxnumerical_features = []
        # categorical_features = list(set(full_train.columns) - set(numerical_features))
        # categorical_features = ["year", "stage", "home", "away", "weekday", "hour"]
        categorical_features = [
            "num_tv",
            "year",
            "stage",
            "home",
            "away",
            "weekday",
            "hour",
            # "month",
            # "stadium",
            # "home_rank",
        ]

        numerical_transformer.fit(full_train[numerical_features])
        categorical_transformer.fit(full_train[categorical_features])

        col_transformers = ColumnTransformer(
            transformers=[
                # (名前, 変換器, 対象列) のタプルで指定
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            # もし残したい列があれば remainder='passthrough' を指定。
            remainder="passthrough",
        )
        col_transformers.fit(full_train)

        col_transformers.set_output(transform="pandas")
        train_df = col_transformers.transform(full_train)
        test_df = col_transformers.transform(full_test)
        # print(f"{train_df.columns[0:20]=}, {test_df.shape=}")
        # print(f"{train_df.columns[20:40]=}, {test_df.shape=}")
        # print(f"{train_df.columns[40:60]=}, {test_df.shape=}")
        # print(f"{train_df.columns[60:80]=}, {test_df.shape=}")
        # print(f"{train_df.columns[-100:-80]=}, {test_df.shape=}")
        # print(f"{train_df.columns[-80:-60]=}, {test_df.shape=}")
        # print(f"{train_df.columns[-60:-40]=}, {test_df.shape=}")
        # print(f"{train_df.columns[-40:]=}, {test_df.shape=}")

        self.data = self.convert_np_to_torch(train_df)
        self.label_scaler = StandardScaler()
        # self.label_scaler.fit(stadium[["capa"]])
        self.label = self.label_scaler.fit_transform(
            (self.raw_label).values.reshape(-1, 1)
            # (self.raw_label - self.train_mean).values.reshape(-1, 1)
        )
        # print(f"{self.label=}")
        # print(f"{self.label.shape=}")
        self.label = torch.tensor(self.label, dtype=torch.float32).flatten()
        self.label = self.label.unsqueeze(1)  # Add a dimension for label

        self.label_number = 1
        self.test_data = self.convert_np_to_torch(test_df)

        data_to_save = {
            "data": self.data,
            "label": self.label,
            "test_data": self.test_data,
            "id": self.test_id,
            "raw_label": self.raw_label,
            "target_feature": self.target_feature,
        }
        joblib.dump(data_to_save, self.PREPOCESSED_PATH)
        print(f"Preprocessed dataset saved to {self.PREPOCESSED_PATH}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_name(self):
        return self.dateset_name

    def get_column_number(self):
        return self.data.shape[1]

    def get_label_number(self):
        return self.label_number

    def get_train_dataset(self):
        (train_data, train_label, test_data) = self.get_numpy_data()
        return SimpleDataset(
            train_data,
            train_label,
            0,
            self.dateset_name + "_train" + str(0 + 1),
        )

    def get_test_dataset(self):
        (train_data, train_label, test_data) = self.get_numpy_data()
        return SimpleDataset(
            test_data,
            None,
            0,
            self.dateset_name + "_train" + str(0 + 1),
        )
