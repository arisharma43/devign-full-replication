# -*- coding: utf-8 -*-
"""
This module is intended to join all the pipeline in separated tasks
to be executed individually or in a flow by using command-line options

Example:
Dataset embedding and processing:
    $ python taskflows.py -e -pS
"""

import argparse
import gc
import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch

from gensim.models.word2vec import Word2Vec

import configs
import src.data as data
import src.prepare as prepare
import src.process as process
import src.utils.functions.cpg as cpg

PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select(dataset, context):
    result = dataset
    if context.projects:
        result = result.loc[result["project"].isin(context.projects)]

    max_len = context.max_func_length
    if max_len is not None:
        result = result.loc[result.func.str.len() <= max_len]

    limit_per_project = context.limit_per_project
    if limit_per_project is not None:
        result = (
            result.sort_values(["project", "func"], kind="mergesort")
            .groupby("project", group_keys=False)
            .head(limit_per_project)
        )

    max_rows = context.max_rows
    if max_rows is not None:
        result = result.head(max_rows)

    return result


def create_task():
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)
    filtered = data.apply_filter(raw, lambda df: select(df, context))
    filtered = data.clean(filtered)
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = prepare.joern_parse(
            context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}"
        )
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)
    # Create CPG with graphs json files
    json_files = prepare.joern_create(
        context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files
    )
    for (s, slice), json_file in zip(slices, json_files):
        graphs = prepare.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()


def embed_task():
    context = configs.Embed()
    # Tokenize source code into tokens
    dataset_files = data.get_directory_files(PATHS.cpg)
    w2vmodel = Word2Vec(**context.w2v_args)
    w2v_init = True
    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        tokens_dataset = data.tokenize(cpg_dataset)
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")
        tokens_list = [tokens for tokens in tokens_dataset.tokens.tolist() if tokens]
        if not tokens_list:
            print(f"Skipping {file_name}: no tokens extracted.")
            continue
        # word2vec used to learn the initial embedding of each token
        w2vmodel.build_vocab(corpus_iterable=tokens_list, update=not w2v_init)
        w2vmodel.train(
            corpus_iterable=tokens_list,
            total_examples=w2vmodel.corpus_count,
            epochs=1,
        )
        if w2v_init:
            w2v_init = False
        # Embed cpg to node representation and pass to graph data structure
        cpg_dataset["nodes"] = cpg_dataset.apply(
            lambda row: cpg.parse_to_nodes(row.cpg, context.nodes_dim), axis=1
        )
        # remove rows with no nodes
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        cpg_dataset["input"] = cpg_dataset.apply(
            lambda row: prepare.nodes_to_input(
                row.nodes, row.target, context.nodes_dim, w2vmodel.wv, context.edge_type
            ),
            axis=1,
        )
        data.drop(cpg_dataset, ["nodes"])
        print(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        data.write(
            cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}"
        )
        del cpg_dataset
        gc.collect()
    print("Saving w2vmodel.")
    w2vmodel.save(f"{PATHS.w2v}/{FILES.w2v}")


def process_task(stopping):
    context = configs.Process()
    devign = configs.Devign()
    set_seed(context.seed)
    model_path = PATHS.model + FILES.model
    input_dataset = data.loads(PATHS.input, ratio=context.dataset_ratio)
    if context.shuffle:
        input_dataset = input_dataset.sample(
            frac=1.0, random_state=context.seed
        ).reset_index(drop=True)
    class_counts = input_dataset["target"].value_counts()
    negative = float(class_counts.get(0, 0))
    positive = float(class_counts.get(1, 0))
    pos_weight = negative / positive if positive > 0 else 1.0
    if context.balance_classes and positive > 0:
        # When we rebalance batches via sampling, keep BCE weights neutral
        pos_weight = 1.0
    os.makedirs(PATHS.model, exist_ok=True)
    model = process.Devign(
        path=model_path,
        device=DEVICE,
        model=devign.model,
        learning_rate=devign.learning_rate,
        weight_decay=devign.weight_decay,
        loss_lambda=devign.loss_lambda,
        pos_weight=pos_weight,
        scheduler_cfg=devign.scheduler,
    )
    train = process.Train(model, context.epochs)
    # split the dataset and pass to DataLoader with batch size
    train_ds, val_ds, test_ds = data.train_val_test_split(
        input_dataset,
        val_size=context.val_ratio,
        test_size=context.test_ratio,
        shuffle=context.shuffle,
        random_state=context.seed,
    )
    train_loader = train_ds.get_loader(
        context.batch_size,
        shuffle=context.shuffle and not context.balance_classes,
        balance=context.balance_classes,
    )
    val_loader = val_ds.get_loader(context.batch_size, shuffle=False)
    test_loader = test_ds.get_loader(context.batch_size, shuffle=False)
    train_loader_step = process.LoaderStep("Train", train_loader, DEVICE)
    val_loader_step = process.LoaderStep("Validation", val_loader, DEVICE)
    test_loader_step = process.LoaderStep("Test", test_loader, DEVICE)

    if stopping:
        early_stopping = process.EarlyStopping(model, patience=context.patience)
        train(train_loader_step, val_loader_step, early_stopping)
        model.load()
    else:
        train(train_loader_step, val_loader_step)
        model.save()

    process.predict(model, test_loader_step)


def main():
    """
    main function that executes tasks based on command-line options
    """
    parser: ArgumentParser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--prepare', help='Prepare task', required=False)
    parser.add_argument("-c", "--create", action="store_true")
    parser.add_argument("-e", "--embed", action="store_true")
    parser.add_argument("-p", "--process", action="store_true")
    parser.add_argument("-pS", "--process_stopping", action="store_true")

    args = parser.parse_args()

    if args.create:
        create_task()
    if args.embed:
        embed_task()
    if args.process:
        process_task(False)
    if args.process_stopping:
        process_task(True)


if __name__ == "__main__":
    main()
