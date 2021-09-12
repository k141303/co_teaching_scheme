import os
import glob

import torch
from torch.utils.data import DataLoader, Dataset

import hydra

from data.file_utils import FileUtils
from data.datasets.array_utils import ArrayTools
from data.datasets.settings import *
from data.datasets.iob2tagging_results_dataset import ShinraResultsDataset, ShinraDataset

class ShinraSelfTrainingDataset(ShinraResultsDataset):
    def __init__(self, *args, prediction_labels=None, **kwargs):
        super().__init__(*args, ignr_labels=True, **kwargs)
        self.prediction_data = {}
        for d in FileUtils.JsonL.load(hydra.utils.to_absolute_path(prediction_labels)):
            self.prediction_data[(d["page_id"], d["part_id"])] = d["iob2tag"]
        self.class_counts = self.labeled_class_counts

    def __getitem__(self, i):
        d = self.data[i]
        labeled_d = self.labeled_data.get((d["pageid"], d["part_id"]))
        if labeled_d is not None:
            d["labels"] = torch.LongTensor(labeled_d)
        else:
            d["labels"] = torch.LongTensor(self.prediction_data[(d["pageid"], d["part_id"])])
        d["input_ids"] = torch.LongTensor(d["input_ids"])
        d["attention_mask"] = torch.LongTensor(d["attention_mask"])
        d["offsets"] = torch.LongTensor(d["offsets"])
        return d

def load_shinra_self_training_dataset(cfg):
    system_results_dir = hydra.utils.to_absolute_path(cfg.data.system_results_dir)
    file_paths = sorted(glob.glob(
        os.path.join(system_results_dir, "annotations", "*")
    ))
    systems = FileUtils.Json.load(
        os.path.join(system_results_dir, "systems.json")
    )

    train_dir = hydra.utils.to_absolute_path(cfg.data.train_dir)
    attributes = FileUtils.Json.load(
        os.path.join(train_dir, "attributes.json")
    )
    labeled_file_paths = sorted(glob.glob(
        os.path.join(train_dir, "annotations", "*")
    ))
    ext_pageid = lambda x:int(os.path.splitext(os.path.basename(x))[0])
    labeled_file_paths = {ext_pageid(file_path): file_path for file_path in labeled_file_paths}

    dev_pageids = set(map(int,FileUtils.Json.load(
        hydra.utils.to_absolute_path(cfg.data.dev_pageids)
    )))

    dev_file_paths = [file_path for pageid, file_path in labeled_file_paths.items() if pageid in dev_pageids]

    train_file_paths = [(file_path, labeled_file_paths.get(ext_pageid(file_path)))
        for file_path in file_paths if ext_pageid(file_path) not in dev_pageids]

    if cfg.data.debug:
        train_file_paths = train_file_paths[:20]

    train_dataset = ShinraSelfTrainingDataset(
        train_file_paths,
        attributes,
        systems,
        seq_len=cfg.data.seq_len,
        duplicate=cfg.data.duplicate,
        prediction_labels=cfg.data.prediction_labels
    )
    dev_dataset = ShinraDataset(
        dev_file_paths,
        attributes,
        seq_len=cfg.data.seq_len,
        duplicate=cfg.data.duplicate
    )
    return train_dataset, dev_dataset
