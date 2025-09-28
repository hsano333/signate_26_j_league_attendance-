class ModelInfo:
    def __init__(
        self, model, model_name, dataset, task, directory, eval, transform_label, config
    ):
        self.model = model
        self.model_name = model_name
        self.dataset = dataset
        self.task = task
        self.directory = directory
        self.evaluate = eval
        self.transform_label = transform_label
        self.config = config
