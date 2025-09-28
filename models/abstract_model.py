from abc import ABC, abstractmethod
from dataset.src.combined_dataset import CombinedDataset


class IModel(ABC):
    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def learn(self, params=None):
        pass

    @abstractmethod
    def reload(self):
        pass

    @abstractmethod
    def forecast(self, data) -> CombinedDataset:
        pass

    @abstractmethod
    def get_model_params(self, trial, model_config):
        pass

    @abstractmethod
    def get_model_optimized_params(self, model_config):
        pass

    @abstractmethod
    def check_params(self, params):
        pass
