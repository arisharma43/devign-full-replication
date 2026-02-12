import glob

import pandas as pd
import numpy as np
import os
import src.utils.functions.parse as parse

from os import listdir
from os.path import isfile, join
from src.utils.objects.input_dataset import InputDataset
from sklearn.model_selection import train_test_split


def read(path, json_file):
    """
    :param path: str
    :param json_file: str
    :return DataFrame
    """
    return pd.read_json(os.path.join(path, json_file))


def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]


def load(path, pickle_file, ratio=1):
    dataset = pd.read_pickle(os.path.join(path, pickle_file))
    dataset.info(memory_usage="deep")
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)

    return dataset


def write(data_frame: pd.DataFrame, path, file_name):
    data_frame.to_pickle(os.path.join(path, file_name))


def apply_filter(data_frame: pd.DataFrame, filter_func):
    return filter_func(data_frame)


def rename(data_frame: pd.DataFrame, old, new):
    return data_frame.rename(columns={old: new})


def tokenize(data_frame: pd.DataFrame):
    data_frame.func = data_frame.func.apply(parse.tokenizer)
    # Change column name
    data_frame = rename(data_frame, "func", "tokens")
    # Keep just the tokens
    return data_frame[["tokens"]]


def to_files(data_frame: pd.DataFrame, out_path):
    # path = f"{self.out_path}/{self.dataset_name}/"
    os.makedirs(out_path, exist_ok=True)

    for idx, row in data_frame.iterrows():
        file_name = f"{idx}.c"
        with open(os.path.join(out_path, file_name), "w") as f:
            f.write(row.func)


def create_with_index(data, columns):
    data_frame = pd.DataFrame(data, columns=columns)
    data_frame.index = list(data_frame["Index"])

    return data_frame


def inner_join_by_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


def train_val_test_split(
    data_frame: pd.DataFrame,
    val_size=0.1,
    test_size=0.1,
    shuffle=True,
    random_state=None,
):
    print("Splitting Dataset")

    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be >=0 and sum to < 1")

    stratify = data_frame.target if data_frame.target.nunique() > 1 else None

    if val_size + test_size == 0:
        train_df = data_frame.reset_index(drop=True)
        empty = pd.DataFrame(columns=train_df.columns)
        return InputDataset(train_df), InputDataset(empty), InputDataset(empty)

    train_df, temp_df = train_test_split(
        data_frame,
        test_size=val_size + test_size,
        shuffle=shuffle,
        stratify=stratify,
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)

    if val_size == 0:
        val_df = pd.DataFrame(columns=train_df.columns)
        test_df = temp_df.reset_index(drop=True)
    elif test_size == 0:
        val_df = temp_df.reset_index(drop=True)
        test_df = pd.DataFrame(columns=train_df.columns)
    else:
        stratify_temp = temp_df.target if stratify is not None else None
        test_fraction = test_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_fraction,
            shuffle=shuffle,
            stratify=stratify_temp,
            random_state=random_state,
        )
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return InputDataset(train_df), InputDataset(val_df), InputDataset(test_df)


def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]


def loads(data_sets_dir, ratio=1):
    data_sets_files = sorted(
        [f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f))]
    )

    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)

    datasets = [load(data_sets_dir, ds_file) for ds_file in data_sets_files]
    if not datasets:
        raise ValueError(f"No datasets found in {data_sets_dir}")

    return pd.concat(datasets, ignore_index=True)


def clean(data_frame: pd.DataFrame):
    return data_frame.drop_duplicates(subset="func", keep=False)


def drop(data_frame: pd.DataFrame, keys):
    for key in keys:
        del data_frame[key]


def slice_frame(data_frame: pd.DataFrame, size: int):
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)
