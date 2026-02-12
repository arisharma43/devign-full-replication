import json
import torch


class Config(object):
    def __init__(self, config, file_path="configs.json"):
        with open(file_path) as config_file:
            self._config = json.load(config_file)
            self._config = self._config.get(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_property(self, property_name):
        return self._config.get(property_name)

    def get_device(self):
        return self.device

    def all(self):
        return self._config


class Create(Config):
    def __init__(self):
        super().__init__("create")

    @property
    def projects(self):
        return self.get_property("projects") or []

    @property
    def slice_size(self):
        return self.get_property("slice_size")

    @property
    def joern_cli_dir(self):
        return self.get_property("joern_cli_dir")

    @property
    def max_func_length(self):
        return self.get_property("max_func_length")

    @property
    def limit_per_project(self):
        return self.get_property("limit_per_project")

    @property
    def max_rows(self):
        return self.get_property("max_rows")


class Data(Config):
    def __init__(self, config):
        super().__init__(config)

    @property
    def cpg(self):
        return self.get_property("cpg")

    @property
    def raw(self):
        return self.get_property("raw")

    @property
    def input(self):
        return self.get_property("input")

    @property
    def model(self):
        return self.get_property("model")

    @property
    def tokens(self):
        return self.get_property("tokens")

    @property
    def w2v(self):
        return self.get_property("w2v")


class Paths(Data):
    def __init__(self):
        super().__init__("paths")

    @property
    def joern(self):
        return self.get_property("joern")


class Files(Data):
    def __init__(self):
        super().__init__("files")

    @property
    def tokens(self):
        return self.get_property("tokens")

    @property
    def w2v(self):
        return self.get_property("w2v")


class Embed(Config):
    def __init__(self):
        super().__init__("embed")

    @property
    def nodes_dim(self):
        return self.get_property("nodes_dim")

    @property
    def w2v_args(self):
        return self.get_property("word2vec_args")

    @property
    def edge_type(self):
        return self.get_property("edge_type")


class Process(Config):
    def __init__(self):
        super().__init__("process")

    @property
    def epochs(self):
        return self.get_property("epochs")

    @property
    def patience(self):
        return self.get_property("patience")

    @property
    def batch_size(self):
        return self.get_property("batch_size")

    @property
    def dataset_ratio(self):
        return self.get_property("dataset_ratio")

    @property
    def shuffle(self):
        return self.get_property("shuffle")

    @property
    def val_ratio(self):
        return self.get_property("val_ratio")

    @property
    def test_ratio(self):
        return self.get_property("test_ratio")

    @property
    def seed(self):
        return self.get_property("seed")

    @property
    def balance_classes(self):
        return self.get_property("balance_classes")


class Devign(Config):
    def __init__(self):
        super().__init__("devign")

    @property
    def learning_rate(self):
        return self.get_property("learning_rate")

    @property
    def weight_decay(self):
        return self.get_property("weight_decay")

    @property
    def loss_lambda(self):
        return self.get_property("loss_lambda")

    @property
    def model(self):
        return self.get_property("model")

    @property
    def scheduler(self):
        return self.get_property("scheduler") or {}
