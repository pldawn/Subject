import json
import math
import os
from abc import ABC
from typing import Iterator
import torch
from torch.utils import data
from torch.utils.data import IterableDataset, TensorDataset
from torch.utils.data.dataset import T_co


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
                    batch_size: int = 1,
                    shuffle: bool = False,
                    num_workers: int = 0,
                    device=None):
        # 得到数据集路径和config
        train_path = os.path.join(corpus_tag_or_dir, "train.json")
        eval_path = os.path.join(corpus_tag_or_dir, "eval.json")
        config = json.load(open(os.path.join(corpus_tag_or_dir, "config.json")))
        config["batch_size"] = batch_size
        config["shuffle"] = shuffle
        config["num_workers"] = num_workers

        # 实例化data loader
        train_loader = DataLoader.from_file(train_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        if os.path.exists(eval_path):
            eval_loader = DataLoader.from_file(eval_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            eval_loader = None

        # 定义map_func
        def map_func(one_batch):
            return map_func_for_graph(
                one_batch,
                config["labels_map"],
                config["entity_map"],
                config["relation_map"],
                config["meta_entity_map"],
                config["meta_relation_map"]
            )

        # 生成loader
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


def map_func_for_graph(one_batch, labels_map, entity_map, relation_map, meta_entity_map, meta_relation_map):
    one_batch = [json.loads(item.strip()) for item in one_batch]

    keys = one_batch[0].keys()
    cache_batch = {}

    assert "inputs" in keys, "corpus must contain key: 'inputs'."

    if "labels" in keys:
        assert labels_map is not None, "if corpus has labels, labels_map can't be None."

    # 遍历所有sample，将按样本区分的one batch转为按key区分的one batch
    for item in one_batch:
        for key in keys:

            # 使用labels_map编码labels
            if key == "labels":
                encoded_label = labels_map[item[key]]
                cache_batch.setdefault(key, []).append(encoded_label)

            elif key == "head_ids":
                cache_batch.setdefault(key, []).append(entity_map[item[key]])

            elif key == "relation_ids":
                cache_batch.setdefault(key, []).append(relation_map[item[key]])

            elif key == "tail_ids":
                cache_batch.setdefault(key, []).append(entity_map[item[key]])

            elif key == "meta_head_ids":
                cache_batch.setdefault(key, []).append(meta_entity_map[item[key]])

            elif key == "meta_relation_ids":
                cache_batch.setdefault(key, []).append(meta_relation_map[item[key]])

            elif key == "meta_tail_ids":
                cache_batch.setdefault(key, []).append(meta_entity_map[item[key]])

    # 组装新的one batch
    new_batch = cache_batch

    # 转为tensor
    for key in new_batch.keys():
        new_batch[key] = torch.tensor(new_batch[key])

    return new_batch
