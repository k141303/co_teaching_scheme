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
from data.datasets.settings import *

class ShinraDataset(Dataset):
    def __init__(
            self,
            file_paths,
            attributes,
            seq_len=512,
            duplicate=50
        ):
        self.data_config = {
            "seq_len": seq_len,
            "duplicate": duplicate
        }

        data = self._load(file_paths)
        self.pageids = sorted(set([d["pageid"] for d in data]))
        self.attributes = attributes
        self.data, self.answers, self.class_counts, self.max_min_num_span_tokens = self._preprocess(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        d["input_ids"] = torch.LongTensor(d["input_ids"])
        d["attention_mask"] = torch.LongTensor(d["attention_mask"])
        d["labels"] = torch.LongTensor(d["labels"])
        d["offsets"] = torch.LongTensor(d["offsets"])
        return d

    def save_pageids(self, output_path):
        FileUtils.Json.save(output_path, self.pageids)

    @staticmethod
    def _mp_load(file_paths):
        data = list(map(FileUtils.Json.load, file_paths))

        ext_pageid = lambda x:os.path.splitext(os.path.basename(x))[0]
        pageids = map(ext_pageid, file_paths)

        for pageid, d in zip(pageids, data):
            d["pageid"] = pageid
        return data

    @classmethod
    def _load(cls, file_paths):
        tasks = np.array_split(file_paths, multi.cpu_count())

        data = []
        with Pool() as p:
            with Pool(multi.cpu_count()) as p:
                for mp_data in p.imap(cls._mp_load, tasks):
                    data += mp_data

        return data

    @staticmethod
    def _get_offsets(tokens):
        # Calc chalacter-level offsets
        offsets = []
        for i, lines in enumerate(tokens):
            offsets.append([])
            for token in lines:
                l = len(re.sub("@@$", "", token))
                s = 0 if len(offsets[i]) == 0 else offsets[i][-1][-1]
                e = s + l
                offsets[i].append((i, s, e))
        return offsets

    @staticmethod
    def add_newline_token(tokens, token_ids, offsets, token="<s>"):
        for i in range(len(tokens)):
            if i == len(tokens) - 1 or len(tokens[i]) == 0:
                continue
            offsets[i].append(offsets[i][-1])
            tokens[i].append(token)
            token_ids[i].append(CLS_TOKEN_ID)
        return tokens, token_ids, offsets

    @staticmethod
    def _mp_preprocess(inputs):
        data, config, attributes = inputs

        processed_data = []
        answers = defaultdict(lambda:defaultdict(set))
        iob2cnt = Counter()
        all_num_span_tokens = defaultdict(list)
        for d in data:
            # Calc chalacter-level offsets
            offsets = ShinraDataset._get_offsets(d["tokens"])
            d["tokens"], d["token_ids"], offsets = ShinraDataset.add_newline_token(d["tokens"], d["token_ids"], offsets)

            # Offset => IOB2tag
            iob2tags = [[TAG_O] * len(d["tokens"][i]) for i in range(len(d["tokens"]))]
            attr_iob2tags = {attr:copy(iob2tags)  for attr in attributes}
            for ann in d["annotations"]:
                attribute = ann["attribute"]
                start, end = ann["token_offset"]["start"], ann["token_offset"]["end"]

                answers[d["pageid"]][attribute].add((
                    ann["text_offset"]["start"]["line_id"],
                    ann["text_offset"]["start"]["offset"],
                    ann["text_offset"]["end"]["line_id"],
                    ann["text_offset"]["end"]["offset"]
                ))

                if start["line_id"] == end["line_id"]:
                    for j in range(start["offset"], end["offset"]):
                        attr_iob2tags[attribute][start["line_id"]][j] = TAG_B if j == start["offset"] else TAG_I
                    continue
                for i in range(start["line_id"], end["line_id"]+1):
                    if i == start["line_id"]:
                        for j in range(start["offset"], len(attr_iob2tags[attribute][i])):
                            attr_iob2tags[attribute][i][j] == TAG_B if j == start["offset"] else TAG_I
                    elif i == end["line_id"]:
                        for j in range(0, end["offset"]):
                            attr_iob2tags[attribute][i][j] == TAG_I
                    else:
                        for j in range(0, len(attr_iob2tags[attribute][i])):
                            attr_iob2tags[attribute][i][j] == TAG_I

            # Remove irrelevant contexts
            for attr in attributes:
                if any(map(any, attr_iob2tags[attr][:REMOVE_HEAD])):
                    print(f"The deleted range(head) has annotations of {attr}.")
                if any(map(any, attr_iob2tags[attr][-REMOVE_TAIL:])):
                    print(f"The deleted range(tail) has annotations of {attr}.")
            attr_iob2tags = {attr: iob2tags[REMOVE_HEAD:-REMOVE_TAIL] for attr, iob2tags in attr_iob2tags.items()}
            offsets = offsets[REMOVE_HEAD:-REMOVE_TAIL]
            tokens = d["tokens"][REMOVE_HEAD:-REMOVE_TAIL]
            token_ids = d["token_ids"][REMOVE_HEAD:-REMOVE_TAIL]

            # Flatten
            attr_iob2tags = {attr: ArrayTools.flatten(iob2tags) for attr, iob2tags in attr_iob2tags.items()}
            attr_iob2tags = [attr_iob2tags[attr] for attr in attributes]
            attr_iob2tags = [*zip(*attr_iob2tags)]
            offsets = ArrayTools.flatten(offsets)
            tokens = ArrayTools.flatten(tokens)
            token_ids = ArrayTools.flatten(token_ids)

            zip_array = [*zip(offsets, tokens, token_ids, attr_iob2tags)]
            for i in reversed(range(len(zip_array))):
                if i == 0 or zip_array[i][1] != "▁" or zip_array[i-1][1] != "▁":
                    continue
                del zip_array[i]

            offsets, _, _, attr_iob2tags = map(list, zip(*zip_array))
            for attr, iob2tag in zip(attributes, zip(*attr_iob2tags)):
                start = None
                for i, tag in enumerate(iob2tag):
                    if start is not None and tag != TAG_I:
                        all_num_span_tokens[attr].append(i - start)
                        start = None
                    if tag == TAG_B:
                        start = i
                if start is not None:
                    all_num_span_tokens[attr].append(len(iob2tag) - start)

            parts = ArrayTools.slide(zip_array, config["seq_len"]-2, config["duplicate"])
            for part_id, part in enumerate(parts):
                part_offsets, _, part_token_ids, part_iob2tags = map(list, zip(*part))

                iob2cnt += Counter(ArrayTools.flatten(part_iob2tags))

                part_token_ids = [CLS_TOKEN_ID] + part_token_ids + [SEP_TOKEN_ID]
                attention_mask = [1] * len(part_token_ids)
                part_iob2tags = [[-100] * len(attributes)] + part_iob2tags
                part_offsets = [(-1, -1, -1)] + part_offsets

                part_token_ids = ArrayTools.padding(part_token_ids, config["seq_len"], PAD_TOKEN_ID)
                attention_mask = ArrayTools.padding(attention_mask, config["seq_len"], 0)
                part_offsets = ArrayTools.padding(part_offsets, config["seq_len"], (-1, -1, -1))
                part_iob2tags = ArrayTools.padding(part_iob2tags, config["seq_len"], [-100]*len(attributes))
                d = {
                    "pageid": d["pageid"],
                    "part_id": part_id,
                    "input_ids": part_token_ids,
                    "attention_mask": attention_mask,
                    "labels": part_iob2tags,
                    "offsets": part_offsets
                }
                processed_data.append(d)

        return processed_data, dict(answers), iob2cnt, all_num_span_tokens

    def _preprocess(self, data):
        tasks = np.array_split(data, multi.cpu_count()*5)
        tasks = [(t, self.data_config, self.attributes) for t in tasks]

        processed_data = []
        answers = {}
        iob2cnt = Counter()
        all_num_span_tokens = defaultdict(list)
        with Pool(multi.cpu_count()) as p, tqdm.tqdm(total=len(tasks)) as t:
            for _processed_data, _answers, _iob2cnt, _all_num_span_tokens in p.imap(self._mp_preprocess, tasks):
                processed_data += _processed_data
                iob2cnt += _iob2cnt
                answers.update(_answers)
                for attr in _all_num_span_tokens:
                    all_num_span_tokens[attr] += _all_num_span_tokens[attr]
                t.update()

        max_min_num_span_tokens = {}
        for attr, array in all_num_span_tokens.items():
            max_min_num_span_tokens[attr] = (max(array), min(array))

        return processed_data, answers, iob2cnt, max_min_num_span_tokens

    @staticmethod
    def _iob2_to_offset(iob2tag, offset):
        start = None
        output = set()
        for i in range(len(iob2tag)):
            if (iob2tag[i] == TAG_O or iob2tag[i] == TAG_B) and start is not None:
                output.add((start[0], start[1], offset[i-1][0], offset[i-1][2]))
                start = None
            if iob2tag[i] == TAG_B:
                start = (offset[i][0], offset[i][1])
        if start is not None:
            output.add((start[0], start[1], offset[i-1][0], offset[i-1][2]))
        return output

    @staticmethod
    def _mp_org_outputs(inputs):
        page_outputs, data_config, attributes = inputs
        page_outputs = dict(page_outputs)

        head = math.ceil(data_config["duplicate"]/2)
        tail = math.floor(data_config["duplicate"]/2)
        for pageid in page_outputs:
            for part_id in range(len(page_outputs[pageid])):
                offset, output = page_outputs[pageid][part_id]
                zip_array = zip(offset, output)
                output = [*filter(lambda x:x[0][0] >= 0, zip_array)]
                if part_id == 0:
                    page_outputs[pageid][part_id] = output[:-tail]
                elif part_id == len(page_outputs[pageid]) -1:
                    page_outputs[pageid][part_id] = output[head:]
                else:
                    page_outputs[pageid][part_id] = output[head:-tail]

        org_outputs = {}
        for pageid in page_outputs:
            page_outputs[pageid] = ArrayTools.flatten(page_outputs[pageid].values())
            offset, output = zip(*page_outputs[pageid])
            output = dict(zip(attributes, np.array(output).T))
            org_outputs[pageid] = {attr: ShinraDataset._iob2_to_offset(output[attr], offset) for attr in output}

        return org_outputs

    def _org_outputs(self, outputs):
        page_outputs = defaultdict(lambda:defaultdict(list))
        for pageid, part_id, offset, output in outputs:
            page_outputs[pageid][part_id] = (offset, output)

        tasks = np.array_split([*page_outputs.items()], 100)
        tasks = [(t, self.data_config, self.attributes) for t in tasks]

        org_outputs = {}
        with tqdm.tqdm(total=len(tasks)) as t:
            for _org_outputs in map(ShinraDataset._mp_org_outputs, tasks):
                org_outputs.update(_org_outputs)
                t.update()

        return org_outputs

    @staticmethod
    def _calc_f1(cnt):
        if cnt["TP"] == 0:
            return {"Prec":0, "Rec":0, "F1":0}
        recall = cnt["TP"] / cnt["TPFN"]
        precision = cnt["TP"] / cnt["TPFP"]
        f1 = 2*recall*precision/(recall+precision)
        return {"Prec":precision, "Rec":recall, "F1":f1}

    def evaluate(self, outputs):
        outputs = [out for out in outputs if out[0] in self.answers]

        outputs = self._org_outputs(outputs)

        cnt = defaultdict(lambda:{"TP":0, "TPFP":0, "TPFN":0})
        cnt["micro_ave"]["TPFN"] += 0
        for pageid in self.answers:
            for attr in self.attributes:
                out = outputs[pageid][attr]
                ans = self.answers[pageid][attr]

                cnt[attr]["TP"] += len(out & ans)
                cnt[attr]["TPFP"] += len(out)
                cnt[attr]["TPFN"] += len(ans)

                cnt["micro_ave"]["TP"] += len(out & ans)
                cnt["micro_ave"]["TPFP"] += len(out)
                cnt["micro_ave"]["TPFN"] += len(ans)

        scores = {attr: self._calc_f1(cnt[attr]) for attr in cnt}
        return scores

def load_shinra_dataset(cfg):
    train_dir = hydra.utils.to_absolute_path(cfg.data.train_dir)
    file_paths = sorted(glob.glob(
        os.path.join(train_dir, "annotations", "*")
    ))
    attributes = FileUtils.Json.load(
        os.path.join(train_dir, "attributes.json")
    )

    if cfg.data.debug:
        file_paths = file_paths[:30]

    train_file_paths, dev_file_paths = ArrayTools.random_split(
        file_paths,
        cfg.data.dev_rate
    )
    print(f"Train size:{len(train_file_paths)} Dev size:{len(dev_file_paths)}")

    train_dataset = ShinraDataset(
        train_file_paths,
        attributes,
        seq_len=cfg.data.seq_len,
        duplicate=cfg.data.duplicate
    )
    dev_dataset = ShinraDataset(
        dev_file_paths,
        attributes,
        seq_len=cfg.data.seq_len,
        duplicate=cfg.data.duplicate
    )
    return train_dataset, dev_dataset
