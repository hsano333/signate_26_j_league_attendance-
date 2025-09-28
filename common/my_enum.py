from enum import IntEnum, auto


class MLTask(IntEnum):
    Regression = auto()  # 回帰
    Classification = auto()  # 分類
    Clustering = auto()  # クラスタリング
    TimeSeries = auto()  # 時系列予測
    DimensionalityReduction = auto()  # 次元削減
    AssociationRuleLearning = auto()  # アソシエーションルール学習
    ReinforcementLearning = auto()  # 強化学習
    AnomalyDetection = auto()  # 異常検知
    NaturalLanguageProcessing = auto()  # 自然言語処理
    ImageRecognition = auto()  # 画像認識


class DatasetPhase(IntEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class NumberSign(IntEnum):
    NEG = 0
    ZERO = 1
    POS = 2
