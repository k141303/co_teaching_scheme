import os
import glob
import tqdm

import random

import re

from collections import defaultdict

from multiprocessing import Pool
import multiprocessing as multi

from data.file_utils import FileUtils
from data.tokenization import JanomeBpeTokenizer

def annotation_mapper(annotations, tokenized_sentences, separator, patch={}):
    match_errors = 0
    mapped_annotations = []
    for ann in annotations:
        start, end = ann["text_offset"]["start"], ann["text_offset"]["end"]

        ann["token_offset"] = {
            "start":{"line_id":start["line_id"]},
            "end":{"line_id":end["line_id"]},
            "text":ann["text_offset"]["text"]
        }

        match_error = False

        tokens = tokenized_sentences[start["line_id"]]
        pos = 0
        for i, token in enumerate(tokens):
            token = re.sub(f"{separator}$", "", token)
            if start["offset"] >= pos and start["offset"] < (pos + len(token)):
                ann["token_offset"]["start"]["offset"] = i
                if pos != start["offset"]:
                    match_error = True
                break
            pos += len(token)

        if ann["token_offset"]["start"].get("offset") is None:
            match_errors += 1
            continue

        tokens = tokenized_sentences[end["line_id"]]
        pos = 0
        for i, token in enumerate(tokens):
            token = re.sub(f"{separator}$", "", token)
            if end["offset"] > pos and end["offset"] <= (pos + len(token)):
                ann["token_offset"]["end"]["offset"] = i + 1
                if (pos + len(token)) != end["offset"]:
                    match_error = True
                break
            pos += len(token)

        if ann["token_offset"]["end"].get("offset") is None:
            key=(int(ann["page_id"]), ann["attribute"], ann["text_offset"]["end"]["line_id"], ann["text_offset"]["end"]["offset"])
            ann["token_offset"]["end"]["offset"] = patch.get(key)

        if ann["token_offset"]["end"].get("offset") is None:
            match_errors += 1
            continue

        del ann["html_offset"]

        match_errors += match_error

        mapped_annotations.append(ann)

    return mapped_annotations, match_errors

class ShinraData(object):
    def __init__(
            self,
            data_dir,
            bpe_codes_path,
            vocab_path,
            num_workers=1
        ):
        self.num_workers = num_workers
        self.bpe_codes_path = bpe_codes_path
        self.vocab_path = vocab_path
        self.annotations = self._load_annotations(data_dir)
        self.attributes = self._ext_attributes(self.annotations)
        self.plain_paths = self._load_plain_paths(data_dir)

    def get_pageids(self):
        return {category: set(anns) for category, anns in self.annotations.items()}

    @staticmethod
    def _preprocess(inputs):
        i, targets, bpe_codes_path, vocab_path, output_dir = inputs

        tokenizer = JanomeBpeTokenizer(bpe_codes_path)
        vocab = FileUtils.Json.load(vocab_path)

        for category, page_id, annotations, file_path in tqdm.tqdm(targets, disable=(i!=0)):
            text = FileUtils.File.load(file_path)
            tokenized_sentences = tokenizer.tokenize(text)

            annotations, match_errors = annotation_mapper(
                annotations,
                tokenized_sentences,
                tokenizer.bpe.separator if tokenizer.bpe is not None else ""
            )

            tokenized_sentences_ids = [[vocab.get(t, vocab["<unk>"]) for t in tokens] for tokens in tokenized_sentences]

            save_path = os.path.join(
                output_dir,
                category,
                "annotations",
                f"{page_id}.json"
            )
            data = {
                "tokens": tokenized_sentences,
                "token_ids": tokenized_sentences_ids,
                "annotations": annotations
            }

            FileUtils.Json.save(save_path, data)

    def preprocess(self, output_dir):
        data = []
        for category in self.annotations:
            os.makedirs(os.path.join(output_dir, category, "annotations"), exist_ok=True)
            for page_id in self.annotations[category]:
                if self.plain_paths[category].get(page_id) is None:
                    continue
                data.append((
                    category,
                    page_id,
                    self.annotations[category][page_id],
                    self.plain_paths[category][page_id]
                ))
        batches = [(i, data[i::self.num_workers], self.bpe_codes_path, self.vocab_path, output_dir) for i in range(self.num_workers)]

        with Pool(self.num_workers) as p, \
            tqdm.tqdm(total=len(batches)) as t:
            for _ in p.imap_unordered(self._preprocess, batches):
                t.update()

        # Save Attribute names
        for category in self.attributes:
            FileUtils.Json.save(
                os.path.join(output_dir, category, "attributes.json"),
                sorted(self.attributes[category])
            )

        # Save Pageid list
        for category in self.annotations:
            FileUtils.Json.save(
                os.path.join(output_dir, category, "pageids.json"),
                sorted(map(int, self.annotations[category].keys()))
            )

    def _load_annotations(self, data_dir):
        file_paths = glob.glob(os.path.join(
                data_dir,
                "annotation/*_dist.json"
        ))
        ext_category_name = lambda x:re.match(".*/(.*?)_dist.json", x).group(1)
        categories = map(ext_category_name, file_paths)

        def org_by_pageid(data):
            org_data = defaultdict(list)
            for d in data:
                page_id = str(d["page_id"])
                org_data[page_id].append(d)
            return org_data

        with Pool(multi.cpu_count()) as p:
            data = p.map(FileUtils.JsonL.load, file_paths)
        data = map(org_by_pageid, data)

        return dict(zip(categories, data))

    def _load_plain_paths(self, data_dir):
        file_paths = glob.glob(os.path.join(
                data_dir,
                "plain/*/*.txt"
            )
        )

        plain_paths = defaultdict(dict)
        for file_path in file_paths:
            *_, category, file_name = file_path.split(os.path.sep)
            page_id = re.match(".*/(.*?).txt", file_path).group(1)
            plain_paths[category][page_id] = file_path

        return plain_paths

    def _ext_attributes(self, annotations):
        attributes = defaultdict(set)
        for category in annotations:
            for pageid in annotations[category]:
                for ann in annotations[category][pageid]:
                    attributes[category].add(ann["attribute"])
        return attributes

class ShirnaSystemData(ShinraData):
    def __init__(
            self,
            data_dir,
            system_result_dir,
            bpe_codes_path,
            vocab_path,
            num_workers=1,
            sampling_size=None,
            random_seed=1234,
            train_pageids=None
        ):
        self.num_workers = num_workers
        self.bpe_codes_path = bpe_codes_path
        self.vocab_path = vocab_path
        self.attributes = []
        self.annotations = self._load_annotations(system_result_dir, sampling_size, random_seed, train_pageids)
        self.systems = self._ext_systems(self.annotations)
        self.plain_paths = self._load_plain_paths(data_dir)

    def _ext_systems(self, annotations):
        systems = defaultdict(set)
        for category in annotations:
            for pageid in annotations[category]:
                for ann in annotations[category][pageid]:
                    systems[category] |= set(ann["system"])
        return systems

    def save_systems(self, output_dir):
        for category in self.systems:
            FileUtils.Json.save(
                os.path.join(output_dir, category, "systems.json"),
                sorted(self.systems[category])
            )

    def _load_annotations(self, data_dir, sampling_size, random_seed, train_pageids={}):
        file_paths = glob.glob(os.path.join(
                data_dir,
                "*.json"
        ))

        # file_paths = [file_path for file_path in file_paths if "Airport" in file_path]

        ext_category_name = lambda x:re.match(".*/(.*?).json", x).group(1)
        categories = [*map(ext_category_name, file_paths)]

        def convert(data):
            converted_data = defaultdict(list)
            for d in data:
                page_id = str(d["page_id"])
                for attribute in d["result"]:
                    for ann in d["result"][attribute]:
                        ann["page_id"] = page_id
                        ann["attribute"] = attribute
                        converted_data[page_id].append(ann)

            return converted_data

        sampled_data = {}
        for i in range(len(file_paths)):
            category = categories[i]

            data = FileUtils.JsonL.load(file_paths[i])

            if sampling_size is None or len(data) <= sampling_size:
                sampled_data[category] = convert(data)
                continue

            train_ids = train_pageids.get(category)
            sampled_data[category] = [d for d in data if d["page_id"] in train_ids]
            remain_data = [d for d in data if d["page_id"] not in train_ids]

            random.seed(random_seed)
            if len(sampled_data[category]) > sampling_size:
                sampled_data[category] = random.sample(sampled_data[category], sampling_size)
                sampled_data[category] = convert(sampled_data[category])
            else:
                remain_sampling_size = sampling_size-len(sampled_data[category])
                sampled_data[category] += random.sample(remain_data, remain_sampling_size)

            sampled_data[category] = convert(sampled_data[category])

        return sampled_data
