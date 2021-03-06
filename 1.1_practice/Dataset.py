import json
import math
import os
from abc import ABC
from typing import Iterator, Optional, Dict
import torch
from torch.utils import data
from torch.utils.data import IterableDataset, TensorDataset
from torch.utils.data.dataset import T_co
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class DataIterator:
    def __init__(self, data_loader, map_func=None, device=None):
        self.data_loader = data_loader
        self.iterator = iter(data_loader)
        self.map_func = map_func

        if isinstance(device, torch.device):
            self.device = device
        elif type(device) == str:
            self.device = torch.device(device)
        elif device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            raise TypeError("device must be str or instance of torch.device.")

    @classmethod
    def from_tensor(cls, tensors, batch_size=1, shuffle=False, num_workers=0, map_func=None, device=None):
        data_loader = DataLoader.from_tensor(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return cls(data_loader, map_func, device)

    @classmethod
    def from_file(cls, filenames, batch_size=1, shuffle=False, num_workers=0, map_func=None, device=None):
        data_loader = DataLoader.from_file(filenames, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return cls(data_loader, map_func, device)

    @classmethod
    def from_corpus(cls,
                    corpus_tag_or_dir: str,
                    tokenizer_path: str,
                    batch_size: int = 1,
                    shuffle: bool = False,
                    num_workers: int = 0,
                    is_split_into_words: Optional[bool] = None,
                    labels_map: Optional[Dict[str, int]] = None,
                    max_length: int = 512,
                    preprocess_fn=None,
                    device=None):
        # ????????????????????????config
        train_path = os.path.join(corpus_tag_or_dir, "train.json")
        eval_path = os.path.join(corpus_tag_or_dir, "eval.json")
        config = json.load(open(os.path.join(corpus_tag_or_dir, "config.json")))
        config["batch_size"] = batch_size
        config["shuffle"] = shuffle
        config["num_workers"] = num_workers

        # ?????????data loader
        train_loader = DataLoader.from_file(train_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        if os.path.exists(eval_path):
            eval_loader = DataLoader.from_file(eval_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            eval_loader = None

        # ??????map_func
        if config["corpus_status"] == "preprocessed":
            if config["task_tag"] == "ner":
                map_func_core = map_func_for_preprocessed_ner
            else:
                map_func_core = map_func_for_preprocessed

            if is_split_into_words is None:
                if config["task_tag"] == "ner":
                    is_split_into_words = True
                else:
                    is_split_into_words = False

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            if labels_map is None:
                labels_map = config.get("labels_map", None)

            def map_func(one_batch):
                return map_func_core(one_batch, tokenizer, is_split_into_words, labels_map, max_length, preprocess_fn)

        elif config["corpus_status"] == "encoded":
            map_func = map_func_for_encoded

        else:
            raise KeyError("corpus status must be preprocessed or encoded.")

        # ??????loader
        data_iter = {"train_iter": cls(train_loader, map_func, device), "config": config}

        if eval_loader is not None:
            data_iter.update({"eval_iter": cls(eval_loader, map_func, device)})

        return data_iter

    def __next__(self):
        try:
            one_batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            one_batch = next(self.iterator)

        if self.map_func is not None:
            one_batch = self.map_func(one_batch)

        for k in one_batch:
            one_batch[k] = one_batch[k].to(self.device)

        return one_batch

    def __iter__(self):
        return self


class DataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    @classmethod
    def from_tensor(cls, tensors, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        for i in range(len(tensors)):
            tensor = tensors[i]
            if not isinstance(tensor, torch.Tensor):
                tensors[i] = torch.Tensor(tensor)

        dataset = TensorDataset(*tensors)

        return cls(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    @classmethod
    def from_file(cls, filenames, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        if isinstance(filenames, str):
            filenames = [filenames]

        dataset = FilesDataset(filenames)

        return cls(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)


class FilesDataset(IterableDataset, ABC):
    def __init__(self, filenames):
        super(FilesDataset, self).__init__()

        if isinstance(filenames, str):
            filenames = [filenames]

        self.filenames = filenames

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            generator = yield_from([open(fn, encoding="utf-8") for fn in self.filenames])
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            per_worker = int(math.ceil(len(self.filenames) / float(num_workers)))
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.filenames))
            generator = yield_from([open(fn, encoding="utf-8") for fn in self.filenames[iter_start: iter_end]])

        return generator


def yield_from(iter_list):
    for it in iter_list:
        yield from it


def map_func_for_preprocessed(one_batch, tokenizer, is_split_into_words, labels_map, max_length, preprocess_fn, noise_fn):
    batch_size = len(one_batch)
    one_batch = [json.loads(item.strip()) for item in one_batch]

    keys = one_batch[0].keys()
    cache_batch = {}

    assert "inputs" in keys, "corpus must contain key: 'inputs'."

    if "labels" in keys:
        assert labels_map is not None, "if corpus has labels, labels_map can't be None."

    # ????????????sample????????????????????????one batch?????????key?????????one batch
    for item in one_batch:
        for key in keys:

            # ??????labels_map??????labels
            if key == "labels":
                if type(item[key]) == str:
                    encoded_label = labels_map[item[key]]
                elif type(item[key]) == list:
                    encoded_label = [labels_map[k] for k in item[key]]
                else:
                    raise TypeError("labels must be str or List[str].")

                cache_batch.setdefault(key, []).append(encoded_label)

            elif key == "inputs":
                preprocessed = item[key]

                # ???????????????
                if preprocess_fn is not None:
                    if type(item[key]) == str:
                        preprocessed = preprocess_fn(item[key])
                    elif type(item[key]) == list:
                        preprocessed = [preprocess_fn(k) for k in item[key]]
                    else:
                        raise TypeError("inputs must be str or List[str].")

                # ????????????
                if noise_fn is not None:
                    if type(item[key]) == str:
                        if type(item[key]) == str:
                            preprocessed = noise_fn(item[key])
                        elif type(item[key]) == list:
                            preprocessed = [noise_fn(k) for k in item[key]]
                        else:
                            raise TypeError("inputs must be str or List[str].")

                cache_batch.setdefault(key, []).append(preprocessed)

    # ??????inputs
    encoded = tokenizer.batch_encode_plus(cache_batch["inputs"], padding=True, is_split_into_words=is_split_into_words,
                                          max_length=max_length, truncation=True)

    # ????????????one batch
    new_batch = {}

    if "labels" in keys:
        new_batch.update({"labels": cache_batch["labels"]})

    for key, value in encoded.items():
        new_batch[key] = value

    # ?????????????????????ignore_index??????labels???inputs??????
    if "labels" in keys:
        if type(new_batch["labels"][0]) == list:
            for i in range(batch_size):
                new_batch["labels"][i] = [0] + new_batch["labels"][i][: max_length - 2] + [0]
                new_batch["labels"][i] += (len(new_batch["input_ids"][i]) - len(new_batch["labels"][i])) * [-100]

    # ??????tensor
    for key in new_batch.keys():
        new_batch[key] = torch.tensor(new_batch[key])

    return new_batch


def map_func_for_preprocessed_ner(one_batch, tokenizer, is_split_into_words, labels_map, max_length, preprocess_fn):
    assert is_split_into_words is True

    batch_size = len(one_batch)
    one_batch = [json.loads(item.strip()) for item in one_batch]

    keys = one_batch[0].keys()
    cache_batch = {}

    assert "inputs" in keys, "corpus must contain key: 'inputs'."

    if "labels" in keys:
        assert labels_map is not None, "if corpus has labels, labels_map can't be None."

    # ????????????sample????????????????????????one batch?????????key?????????one batch
    for item in one_batch:
        for key in keys:

            # ??????labels_map??????labels
            if key == "labels":
                if type(item[key]) == str:
                    encoded_label = labels_map[item[key]]
                elif type(item[key]) == list:
                    encoded_label = [labels_map[k] for k in item[key]]
                else:
                    raise TypeError("labels must be str or List[str].")

                cache_batch.setdefault(key, []).append(encoded_label)

            # ??????preprocess_fn??????inputs
            elif key == "inputs":
                preprocessed = item[key]

                # ???????????????
                if preprocess_fn is not None:
                    if type(item[key]) == list:
                        preprocessed = [preprocess_fn(k) for k in item[key]]
                    else:
                        raise TypeError("inputs must be List[str].")

                cache_batch.setdefault(key, []).append(preprocessed)

    # ??????inputs
    encoded = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    for sm in cache_batch["inputs"]:
        input_id = [cls_id] + tokenizer.convert_tokens_to_ids(sm[:max_length - 2]) + [sep_id]
        token_type_id = [0] * len(input_id)
        attention_mask = [1] * len(input_id)

        encoded["input_ids"].append(torch.tensor(input_id))
        encoded["token_type_ids"].append(torch.tensor(token_type_id))
        encoded["attention_mask"].append(torch.tensor(attention_mask))

    for key in encoded:
        encoded[key] = pad_sequence(encoded[key], batch_first=True, padding_value=0)

    # ????????????one batch
    new_batch = {}

    if "labels" in keys:
        new_batch.update({"labels": cache_batch["labels"]})

    for key, value in encoded.items():
        new_batch[key] = value

    # ?????????????????????ignore_index??????labels???inputs??????
    if "labels" in keys:
        if type(new_batch["labels"][0]) == list:
            for i in range(batch_size):
                new_batch["labels"][i] = [0] + new_batch["labels"][i][: max_length - 2] + [0]
                new_batch["labels"][i] += (len(new_batch["input_ids"][i]) - len(new_batch["labels"][i])) * [-100]

    # ??????tensor
    for key in new_batch.keys():
        if key not in ["input_ids", "token_type_ids", "attention_mask"]:
            new_batch[key] = torch.tensor(new_batch[key])

    return new_batch


def map_func_for_encoded(one_batch):
    one_batch = [json.loads(item.strip()) for item in one_batch]

    keys = one_batch[0].keys()
    cache_batch = {}

    for item in one_batch:
        for key in keys:
            cache_batch.setdefault(key, []).append(item[key])

    new_batch = {}
    for key in keys:
        new_batch[key] = torch.tensor(cache_batch[key])

    return new_batch
