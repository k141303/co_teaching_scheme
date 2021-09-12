import os
import glob
import random

import re

import math

from multiprocessing import Pool
import multiprocessing as multi

from copy import deepcopy as copy
from collections import defaultdict, Counter

import numpy as np

import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

import hydra

from data.file_utils import FileUtils
from data.datasets.array_utils import ArrayTools
from data.datasets.iob2tagging_dataset import ShinraDataset
from data.datasets.settings import *

class ShinraResultsDataset(ShinraDataset):
    def __init__(
            self,
            file_paths,
            attributes,
            systems,
            seq_len = 512,
            duplicate = 50,
            ignr_labels = False,
        ):
        self.attributes = attributes
        self.systems = systems
        self.ignr_labels = ignr_labels
        self.data_config = {
            "seq_len": seq_len,
            "duplicate": duplicate
        }

        try:
            file_paths, labeled_file_paths = zip(*file_paths)
            labeled_file_paths = [*filter(lambda x:x is not None, labeled_file_paths)]
        except ValueError:
            file_paths, labeled_file_paths = file_paths, []

        self.labeled_data, self.labeled_class_counts, self.labeled_dataset = self._load_labeled_data(labeled_file_paths, attributes, seq_len, duplicate)

        data = self._load(file_paths)
        self.data, self.class_counts = self._preprocess(data)
        self.answers = self._ext_answers(data)

        total_class_counts = sum([sum(cnt.values()) for cnt in self.class_counts.values()])
        self.labeled_rate = sum(self.labeled_class_counts.values()) / (total_class_counts / len(self.systems))

    def __getitem__(self, i):
        d = self.data[i]
        labeled_d = self.labeled_data.get((d["pageid"], d["part_id"]))
        if labeled_d is not None:
            d["gold_labels"] = labeled_d
        else:
            d["gold_labels"] = torch.ones(
                self.data_config["seq_len"],
                len(self.attributes)
            ,dtype=torch.long) * -100
        d["input_ids"] = torch.LongTensor(d["input_ids"])
        d["attention_mask"] = torch.LongTensor(d["attention_mask"])
        d["labels"] = torch.LongTensor(d["labels"])
        d["offsets"] = torch.LongTensor(d["offsets"])
        return d

    def _load_labeled_data(self, file_paths, attributes, seq_len, duplicate):
        labeled_dataset = ShinraDataset(
            file_paths,
            attributes,
            seq_len=seq_len,
            duplicate=duplicate,
        )
        class_counts = labeled_dataset.class_counts
        labeled_data = {(d["pageid"], d["part_id"]): d["labels"] for d in labeled_dataset}
        return labeled_data, class_counts, labeled_dataset

    def _ext_answers(self, data):
        answers = defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
        for d in data:
            for ann in d["annotations"]:
                attribute = ann["attribute"]
                for system in ann["system"]:
                    if system not in self.systems:
                        continue
                    answers[system][d["pageid"]][attribute].add((
                        ann["text_offset"]["start"]["line_id"],
                        ann["text_offset"]["start"]["offset"],
                        ann["text_offset"]["end"]["line_id"],
                        ann["text_offset"]["end"]["offset"]
                    ))
        return answers

    @staticmethod
    def _mp_preprocess(inputs):
        data, config, attributes, systems, ignr_labels = inputs

        processed_data = []
        answers = defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
        iob2cnt = defaultdict(Counter)
        for d in data:
            offsets = ShinraDataset._get_offsets(d["tokens"])
            d["tokens"], d["token_ids"], offsets = ShinraDataset.add_newline_token(d["tokens"], d["token_ids"], offsets)

            # Offset => IOB2tag
            iob2tags = [[TAG_O] * len(d["tokens"][i]) for i in range(len(d["tokens"]))]
            attr_iob2tags = {attr:copy(iob2tags)  for attr in attributes}
            system_iob2tags = {system:copy(attr_iob2tags) for system in systems}
            for ann in d["annotations"]:
                for system in ann["system"]:
                    if system not in systems:
                        continue
                    attribute = ann["attribute"]
                    if attribute not in attributes:
                        continue
                    start, end = ann["token_offset"]["start"], ann["token_offset"]["end"]

                    if start["line_id"] == end["line_id"]:
                        for j in range(start["offset"], end["offset"]):
                            system_iob2tags[system][attribute][start["line_id"]][j] = TAG_B if j == start["offset"] else TAG_I
                        continue
                    for i in range(start["line_id"], end["line_id"]+1):
                        if i == start["line_id"]:
                            for j in range(start["offset"], len(system_iob2tags[system][attribute][i])):
                                system_iob2tags[system][attribute][i][j] == TAG_B if j == start["offset"] else TAG_I
                        elif i == end["line_id"]:
                            for j in range(0, end["offset"]):
                                system_iob2tags[system][attribute][i][j] == TAG_I
                        else:
                            for j in range(0, len(system_iob2tags[system][attribute][i])):
                                system_iob2tags[system][attribute][i][j] == TAG_I

            # Remove irrelevant contexts
            for system in systems:
                system_iob2tags[system] = {attr: iob2tags[REMOVE_HEAD:-REMOVE_TAIL] for attr, iob2tags in system_iob2tags[system].items()}
            offsets = offsets[REMOVE_HEAD:-REMOVE_TAIL]
            tokens = d["tokens"][REMOVE_HEAD:-REMOVE_TAIL]
            token_ids = d["token_ids"][REMOVE_HEAD:-REMOVE_TAIL]

            # Flatten
            for system in systems:
                system_iob2tags[system] = {attr: ArrayTools.flatten(iob2tags) for attr, iob2tags in system_iob2tags[system].items()}
                system_iob2tags[system] = [system_iob2tags[system][attr] for attr in attributes]
                system_iob2tags[system] = [*zip(*system_iob2tags[system])]
            system_iob2tags = torch.cat([torch.LongTensor(system_iob2tags[system]) for system in systems], dim=-1).tolist()

            offsets = ArrayTools.flatten(offsets)
            tokens = ArrayTools.flatten(tokens)
            token_ids = ArrayTools.flatten(token_ids)

            zip_array = [*zip(offsets, tokens, token_ids, system_iob2tags)]
            for i in reversed(range(len(zip_array))):
                if i == 0 or zip_array[i][1] != "▁" or zip_array[i-1][1] != "▁":
                    continue
                del zip_array[i]

            parts = ArrayTools.slide(zip_array, config["seq_len"]-2, config["duplicate"])
            for part_id, part in enumerate(parts):
                part_offsets, _, part_token_ids, part_iob2tags = map(list, zip(*part))

                for i, s_part_iob2tags in enumerate(zip(*part_iob2tags)):
                    iob2cnt[i//len(attributes)] += Counter(list(s_part_iob2tags))

                active_tokens = [0] + [1] * len(part_token_ids)
                part_token_ids = [CLS_TOKEN_ID] + part_token_ids
                attention_mask = [1] * len(part_token_ids)
                part_iob2tags = [[-100] * (len(attributes) * len(systems))] + part_iob2tags
                part_offsets = [(-1, -1, -1)] + part_offsets

                part_token_ids = ArrayTools.padding(part_token_ids, config["seq_len"]-1, PAD_TOKEN_ID)
                part_token_ids += [SEP_TOKEN_ID]
                attention_mask = ArrayTools.padding(attention_mask, config["seq_len"]-1, 0)
                attention_mask += [1]
                part_offsets = ArrayTools.padding(part_offsets, config["seq_len"], (-1, -1, -1))
                part_iob2tags = ArrayTools.padding(part_iob2tags, config["seq_len"], [-100]*(len(attributes)*len(systems)))
                active_tokens = ArrayTools.padding(active_tokens, config["seq_len"], 0)
                d = {
                    "pageid": d["pageid"],
                    "part_id": part_id,
                    "input_ids": part_token_ids,
                    "attention_mask": attention_mask,
                    "active_tokens": active_tokens,
                    "offsets": part_offsets
                }
                if not ignr_labels:
                    d["labels"] = part_iob2tags
                processed_data.append(d)

        return processed_data, iob2cnt

    def _preprocess(self, data):
        tasks = np.array_split(data, 200)
        tasks = [(t, self.data_config, self.attributes, self.systems, self.ignr_labels) for t in tasks]

        processed_data = []
        iob2cnt = defaultdict(Counter)
        with Pool(multi.cpu_count()//4) as p, tqdm.tqdm(total=len(tasks)) as t:
            for _processed_data, _iob2cnt in p.imap(ShinraResultsDataset._mp_preprocess, tasks):
                processed_data += _processed_data
                for system, _s_iob2cnt in _iob2cnt.items():
                    iob2cnt[system] += _s_iob2cnt
                t.update()
        return processed_data, iob2cnt

    def _org_outputs(self, outputs):
        page_outputs = defaultdict(lambda:defaultdict(list))
        for pageid, part_id, offset, each_output in outputs:
            for system, output in zip(self.systems, each_output):
                page_outputs[(system, pageid)][part_id] = (offset, output)

        tasks = np.array_split([*page_outputs.items()], 100)
        tasks = [(t, self.data_config, self.attributes) for t in tasks]

        org_outputs = {}
        with tqdm.tqdm(total=len(tasks)) as t:
            for _org_outputs in map(ShinraResultsDataset._mp_org_outputs, tasks):
                org_outputs.update(_org_outputs)
                t.update()

        return org_outputs

    def evaluate(self, outputs):
        outputs = self._org_outputs(outputs)

        cnt = defaultdict(lambda:defaultdict(lambda:{"TP":0, "TPFP":0, "TPFN":0}))
        for system in self.systems:
            for pageid in self.answers[system]:
                for attr in self.attributes:
                    out = outputs[(system, pageid)][attr]
                    ans = self.answers[system][pageid][attr]

                    cnt[system][attr]["TP"] += len(out & ans)
                    cnt[system][attr]["TPFP"] += len(out)
                    cnt[system][attr]["TPFN"] += len(ans)

                    cnt[system]["micro_ave"]["TP"] += len(out & ans)
                    cnt[system]["micro_ave"]["TPFP"] += len(out)
                    cnt[system]["micro_ave"]["TPFN"] += len(ans)

                    cnt["micro_ave"]["micro_ave"]["TP"] += len(out & ans)
                    cnt["micro_ave"]["micro_ave"]["TPFP"] += len(out)
                    cnt["micro_ave"]["micro_ave"]["TPFN"] += len(ans)

        scores = {
            system: {attr: self._calc_f1(cnt[system][attr]) for attr in cnt[system]}
            for system in cnt
        }
        return scores

    @staticmethod
    def _convert_to_submit_form(pageid, attribute, output):
        return {
            "page_id":int(pageid),
            "attribute":attribute,
            "text_offset":{
                "start":{
                    "line_id": output[0],
                    "offset": output[1]
                },
                "end":{
                    "line_id": output[2],
                    "offset": output[3]
                }
            }
        }

    def convert(self, outputs):
        outputs = self._org_outputs(outputs)
        submit_formats = defaultdict(list)
        for system, pageid in outputs:
            for attr in outputs[(system, pageid)]:
                out = outputs[(system, pageid)][attr]
                if len(out) == 0:
                    continue
                for o in out:
                    submit_formats[system].append(
                        self._convert_to_submit_form(pageid, attr, o)
                    )
        return submit_formats

    def convert_single(self, outputs):
        outputs = super()._org_outputs(outputs)
        submit_format = []
        for pageid in outputs:
            for attr in outputs[pageid]:
                out = outputs[pageid][attr]
                if len(out) == 0:
                    continue
                for o in out:
                    submit_format.append(
                        self._convert_to_submit_form(pageid, attr, o)
                    )
        return submit_format

def load_shinra_results_dataset(cfg):
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

    file_paths = [(file_path, labeled_file_paths.get(ext_pageid(file_path))) for file_path in file_paths]
    labeled_file_paths = [*filter(lambda x:x[1] is not None, file_paths)]
    un_labeled_file_paths = [*filter(lambda x:x[1] is None, file_paths)]

    if cfg.data.ignore_train_pageids:
        labeled_file_paths = []

    if cfg.data.contain_all_train_pageids:
        sampled_file_paths = labeled_file_paths
        remain_file_paths = un_labeled_file_paths
        if len(sampled_file_paths) > cfg.data.max_sample_size:
            sampled_file_paths = random.sample(sampled_file_paths, cfg.data.max_sample_size)
    else:
        sampled_file_paths = []
        remain_file_paths = labeled_file_paths + un_labeled_file_paths

    if cfg.data.max_sample_size is None or len(remain_file_paths) <= (cfg.data.max_sample_size - len(sampled_file_paths)):
        sampled_file_paths += remain_file_paths
    else:
        sampled_file_paths += random.sample(remain_file_paths, (cfg.data.max_sample_size - len(sampled_file_paths)))

    if cfg.data.debug:
        sampled_file_paths = sampled_file_paths[:200]

    labeled_sampled_file_paths = [*filter(lambda x:x[1] is not None, sampled_file_paths)]
    un_labeled_sampled_file_paths = [*filter(lambda x:x[1] is None, sampled_file_paths)]

    train_file_paths, dev_file_paths = ArrayTools.random_split(
        labeled_sampled_file_paths,
        cfg.data.dev_rate
    )
    train_file_paths += un_labeled_sampled_file_paths

    """
    _train_file_paths, _dev_file_paths = ArrayTools.random_split(
        un_labeled_sampled_file_paths,
        cfg.data.dev_rate
    )
    train_file_paths += _train_file_paths
    dev_file_paths += _dev_file_paths
    """
    print(f"Train size:{len(train_file_paths)} Dev size:{len(dev_file_paths)}")

    print(f"All Systems:{systems}")
    if cfg.distillation.systems is not None:
        systems = list(map(str, cfg.distillation.systems))
    print(f"Selected Systems:{systems}")

    train_dataset = ShinraResultsDataset(
        train_file_paths,
        attributes,
        systems,
        seq_len=cfg.data.seq_len,
        duplicate=cfg.data.duplicate
    )

    dev_dataset = ShinraResultsDataset(
        dev_file_paths,
        attributes,
        systems,
        seq_len=cfg.data.seq_len,
        duplicate=cfg.data.duplicate
    )
    return train_dataset, dev_dataset

class ShinraTargetDataset(ShinraResultsDataset):
    def __getitem__(self, i):
        d = self.data[i]
        d["input_ids"] = torch.LongTensor(d["input_ids"])
        d["attention_mask"] = torch.LongTensor(d["attention_mask"])
        d["offsets"] = torch.LongTensor(d["offsets"])
        return d


def load_target_dataset(cfg, systems, attributes):
    target_dir = hydra.utils.to_absolute_path(cfg.data.target_dir)
    file_paths = glob.glob(
        os.path.join(target_dir, "annotations", "*")
    )
    if systems is None:
        systems = FileUtils.Json.load(
            os.path.join(target_dir, "systems.json")
        )

    if cfg.data.target_pageids is not None:
        target_data = FileUtils.JsonL.load(hydra.utils.to_absolute_path(cfg.data.target_pageids))
        target_pageids = set(int(d["page_id"]) for d in target_data)
        ext_pageid = lambda x:int(os.path.splitext(os.path.basename(x))[0])
        file_paths = [*filter(lambda x:ext_pageid(x) in target_pageids, file_paths)]

    if cfg.data.debug:
        file_paths = file_paths[:20]

    target_dataset = ShinraTargetDataset(
        file_paths,
        attributes,
        systems,
        seq_len=cfg.data.seq_len,
        duplicate=cfg.data.duplicate,
        ignr_labels=True
    )
    return target_dataset
