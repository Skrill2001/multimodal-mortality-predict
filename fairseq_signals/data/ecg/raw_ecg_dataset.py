import logging
import os
import sys
import ast
import bisect
import re
import pandas as pd

import wfdb
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Optional, Union
from fairseq_signals.data.ecg import augmentations
from fairseq_signals.data.ecg.augmentations import PERTURBATION_CHOICES
from fairseq_signals.dataclass import ChoiceEnum
from tqdm.auto import tqdm

from .. import BaseDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes

BUCKET_CHOICE = ChoiceEnum(["uniform"])

logger = logging.getLogger(__name__)

class RawECGDataset(BaseDataset):
    def __init__(
        self,
        sample_rate,
        perturbation_mode: Optional[List[PERTURBATION_CHOICES]]=None,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        pad_leads=False,
        leads_to_load: Optional[str]=None,
        label=False,
        label_file: str=None,
        filter=False,
        normalize=False,
        mean_path: Optional[str]=None,
        std_path: Optional[str]=None,
        compute_mask_indices=False,
        leads_bucket=None,
        bucket_selection: BUCKET_CHOICE="uniform",
        training=True,
        **kwargs,
    ):
        super().__init__()

        self.training = training

        self.sample_rate = sample_rate
        self.perturbation_mode = perturbation_mode
        self.retain_original = True if perturbation_mode is not None else False

        self.aug_list = []
        if perturbation_mode is not None:
            p = kwargs.pop("p")
            if hasattr(p, "__len__") and len(p) == 1:
                p = list(p) * len(perturbation_mode)
            elif isinstance(p, float):
                p = [p] * len(perturbation_mode)
                
            for aug, prob in zip(perturbation_mode, p):
                self.aug_list.append(
                    augmentations.instantiate_from_name(aug, p=prob, **kwargs)
                )

        # 用来存每一个样本的长度的
        self.sizes = []
        self.max_sample_size = (
             max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.pad_leads = pad_leads
        if leads_to_load is not None:
            leads_to_load = eval(leads_to_load)
            self.leads_to_load = list(map(self.get_lead_index, leads_to_load))
        else:
            self.leads_to_load = list(range(12))

        self.leads_bucket = leads_bucket
        if leads_bucket:
            leads_bucket = eval(leads_bucket)
            self.leads_bucket = list(map(self.get_lead_index, leads_bucket))
        self.bucket_selection = bucket_selection

        assert not (leads_bucket and pad_leads), (
            "Bucketizing multiple leads does not work with lead-padding. "
            "Please check that --pad_leads is unset when using bucketized dataset."
        )

        self.label = label
        self.label_array = None
        if label_file is not None:
            assert label_file.endswith(".npy"), "--label_file should be ended with .npy."
            self.label_array = np.load(label_file)

        self.shuffle = shuffle
        self.filter = filter
        self.normalize = normalize
        if self.normalize:
            assert not (mean_path is None or std_path is None), (
                "Normalizing needs mean and std to be used for z-normalization. "
                "Please check that --mean_path and --std_path are provided. "
            )
            mean = []
            with open(mean_path, "r") as f:
                for m in f.readlines():
                    mean.append(float(m.strip()))
            self.mean = np.array(mean)[:, None]
            std = []
            with open(std_path, "r") as f:
                for s in f.readlines():
                    std.append(float(s.strip()))
            self.std = np.array(std)[:, None]                
        self.compute_mask_indices = compute_mask_indices
        if self.compute_mask_indices:
            self.mask_compute_kwargs = kwargs
            self._features_size_map = {}
            self._C = kwargs["encoder_embed_dim"]
            self._conv_feature_layers = eval(kwargs["conv_feature_layers"])

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    @property
    def apply_perturb(self):
        return self.perturbation_mode is not None

    def get_lead_index(self, lead: Union[int, str]) -> int:
        if isinstance(lead, int):
            return lead
        lead = lead.lower()
        order = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        try:
            index = order.index(lead)
        except ValueError:
            raise ValueError(
                "Please make sure that the lead indicator is correct"
            )
        return index

    def perturb(self, feats):
        if not self.training:
            return feats

        new_feats = feats.clone()
        for aug in self.aug_list:
            new_feats = aug(new_feats)

        return new_feats

    def postprocess(self, feats, curr_sample_rate=None, leads_to_load=None):
        if (
            (self.sample_rate is not None and self.sample_rate > 0)
            and curr_sample_rate != self.sample_rate
        ):
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        leads_to_load = self.leads_to_load if leads_to_load is None else leads_to_load
        feats = feats.float()
        # 只加载某几个的leads
        feats = self.load_specific_leads(feats, leads_to_load=leads_to_load, pad=self.pad_leads)

        if self.filter:
            import neurokit2 as nk
            feats = torch.from_numpy(
                np.stack([nk.ecg_clean(l, sampling_rate=500) for l in feats])
            )
        if self.normalize:
            for l in leads_to_load:
                feats[l] = (feats[l] - self.mean[l]) / self.std[l]

        if self.training and self.apply_perturb:
            feats = self.perturb(feats)

        return feats
    
    # 加载leads_to_load中的数据，但是对于leads_bucket中的lead数据是按照bucket_selection的策略随机选取的
    def load_specific_leads(self, feats, leads_to_load, pad=True):
        if self.leads_bucket:
            leads_bucket = set(self.leads_bucket)
            leads_to_load = set(leads_to_load)
            if not leads_bucket.issubset(leads_to_load):
                raise ValueError(
                    "Please make sure that --leads_bucket is a subset of --leads_to_load."
                )
            
            leads_to_load = list(leads_to_load - leads_bucket)
            leads_to_load.sort()
            if self.bucket_selection == "uniform":
                choice = np.random.choice(self.leads_bucket, size=1)
            else:
                raise Exception("unknown bucket selection " + self.bucket_selection)
            leads_to_load.extend(choice)

        feats = feats[leads_to_load]
        if self.pad_leads:
            if pad:
                padded = torch.zeros((12, feats.size(-1)))
                padded[leads_to_load] = feats
                feats = padded
        
        return feats

    def crop_to_max_size(self, sample, target_size, rand=False):
        dim = sample.dim()
        size = sample.shape[-1]
        diff = size - target_size
        if diff <= 0:
            return sample

        start = 0
        if rand:
            start = np.random.randint(0, diff + 1)
        end = size - diff + start
        if dim == 1:
            return sample[start:end], start, end
        elif dim == 2:
            return sample[:, start:end], start, end
        else:
            raise AssertionError('Check the dimension of the input data')

    def _compute_mask_indices(self, dims, padding_mask):
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks = 2,
                no_overlap = self.mask_compute_kwargs["no_mask_overlap"],
                min_space = self.mask_compute_kwargs["mask_min_space"]
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap = self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space = self.mask_compute_kwargs["mask_channel_min_space"]
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )

        return mask_indices, mask_channel_indices

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value = value)
    
    def collator(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        originals = None
        if self.retain_original and "original" in samples[0]:
            originals = [s["original"] for s in samples]
        sizes = [s.size(-1) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros((len(sources), len(sources[0]), target_size))
        collated_originals = collated_sources.clone() if originals else None
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
                if originals:
                    collated_originals[i] = originals[i]
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((source.shape[0], -diff,), 0.0)], dim=-1
                )
                if originals:
                    collated_originals[i] = torch.cat(
                        [originals[i], originals[i].new_full((originals[i].shape[0], -diff,), 0.0)], dim=-1
                    )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i], start, end = self.crop_to_max_size(source, target_size, rand=True)
                if originals:
                    collated_originals[i] = originals[i][:,start:end]

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.label:
            out["label"] = torch.stack([s["label"] for s in samples])

        if originals:
            out["original"] = collated_originals

        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._buckted_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if self.compute_mask_indices:
            B = input["source"].size(0)
            T = self._get_mask_indices_dims(input["source"].size(-1))
            padding_mask_reshaped = input["padding_mask"].clone()
            extra = padding_mask_reshaped.size(1) % T
            if extra > 0:
                padding_mask_reshaped = padding_mask_reshaped[:,:-extra]
            padding_mask_reshaped = padding_mask_reshaped.view(
                padding_mask_reshaped.size(0), T, -1
            )
            padding_mask_reshaped = padding_mask_reshaped.all(-1)
            input["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
            mask_indices, mask_channel_indices = self._compute_mask_indices(
                (B, T, self._C),
                padding_mask_reshaped
            )
            input["mask_indices"] = mask_indices
            input["mask_channel_indices"] = mask_channel_indices
            out["sample_size"] = mask_indices.sum().item()

        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding = 0, dilation = 1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1 ) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an examples's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    # 这里按长度排序主要也是为了给后面的bucket来使用的    
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            # NOTE: sort according to the size of each sample
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size
                )
            )
            return np.lexsort(order)[::-1]
            # return order[0]
        else:
            return np.arange(len(self))

    # 分桶主要是为了将相似长度的数据放到一个桶中，每次拿数据都在一个桶拿，减少padding的损失
    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the ecg dataset: "
                f"{self.buckets}"
            )

class FileECGDataset(RawECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        num_buckets=0,
        **kwargs
    ):
        super().__init__(
            sample_rate=sample_rate,
            **kwargs
        )

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if self.min_sample_size is not None and sz < self.min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype = np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarraw array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        res = {'id': index}

        ecg = scipy.io.loadmat(path)

        curr_sample_rate = ecg['curr_sample_rate']
        feats = torch.from_numpy(ecg['feats'])
        res["source"] = self.postprocess(feats, curr_sample_rate)
        if self.retain_original:
            res["original"] = feats

        if self.label:
            if self.label_array is not None:
                res["label"] = torch.from_numpy(self.label_array[ecg["idx"].squeeze()])
            else:
                res["label"] = torch.from_numpy(ecg['label'].squeeze(0))

        return res

    def __len__(self):
        return len(self.fnames)

class PathECGDataset(FileECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        load_specific_lead=False,
        **kwargs
    ):
        super().__init__(manifest_path=manifest_path, sample_rate=sample_rate, **kwargs)
    
        self.load_specific_lead = load_specific_lead

    def collator(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}
        
        sources = [s["source"] for s in samples]
        sizes = [s.size(-1) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros((len(sources), len(sources[0]), target_size))
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((source.shape[0], -diff,), 0.0)], dim=-1
                )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i], start, end = self.crop_to_max_size(source, target_size, rand=True)

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if "target_idx" in samples[0]:
            out["target_idx"] = [s["target_idx"] for s in samples]
        if self.label:
            out["label"] = [s["label"] for s in samples]
        if "attribute_id" in samples[0]:
            out["attribute_id"] = [s["attribute_id"] for s in samples]

        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._buckted_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        out["net_input"] = input

        return out

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        res = {"id": index}

        data = scipy.io.loadmat(path)

        feats, _ = wfdb.rdsamp(data["ecg_path"][0])
        feats = torch.from_numpy(feats.T)

        leads_to_load = data["lead"][0] if "lead" in data else self.leads_to_load
        feats = self.postprocess(feats, curr_sample_rate=None, leads_to_load=leads_to_load)

        res["source"] = feats

        if self.label:
            if self.label_array is not None:
                res["label"] = torch.from_numpy(self.label_array[data["idx"].squeeze()])
            else:
                res["label"] = torch.from_numpy(data["label"].squeeze(0))

        if "target_idx" in data:
            res["target_idx"] = torch.from_numpy(data["target_idx"][0])

        if "attribute_id" in data:
            res["attribute_id"] = data["attribute_id"][0]

        return res


time_bins = [0, 6, 12, 24, 48, 72, 168, 336, 672]
# time_bins = [0, 24, 72, 168, 336, 504]

class FinetuneECGDataset(BaseDataset):
    def __init__(
        self,
        split: str="train",
        records_path: str=None,
        data_dir: str=None,
        sample_rate = 500,
        target_size = 2500,
        **kwargs,
    ):
        super().__init__()

        self.split = split
        self.sample_rate = sample_rate
        self.target_size = target_size
        
        if data_dir is not None:
            self.data_dir = data_dir 
        elif os.path.isdir(os.path.join(os.path.dirname(records_path), 'segmented')):
            self.data_dir = os.path.join(os.path.dirname(records_path), 'segmented')
        else:
            raise Exception(f"ERROR: cannot find a valid data_dir !")

        self.ecg_data = {}
        self.pos_cnt = 0
        self.neg_cnt = 0

        with open(records_path, "r") as f:
            # 读取行标题
            f.readline().strip()
            for _, line in enumerate(f):
                items = line.strip().split(",")
                assert len(items) == 5, f"ERROR: There are something wrong when read records at [{line}]"

                # 是所需的数据集划分:
                if items[2] == self.split:
                    ecg_path = os.path.join(self.data_dir, items[1])
                    self.ecg_data[items[0]] = {
                        "segment_0_path": ecg_path.replace('.mat', '_0.mat'),
                        "segment_1_path": ecg_path.replace('.mat', '_1.mat'),
                        "label": int(items[4])
                    }
                    if int(items[4]) == 1:
                        self.pos_cnt += 1
                    elif int(items[4]) == 0:
                        self.neg_cnt += 1
                    else:
                        raise Exception(f"ERROR: There is a unvalid label at {items[1]}, the label is {items[4]}!")

        self.ecg_idx_list = list(self.ecg_data.keys())
        print(f"Loaded [{len(self.ecg_idx_list)}] ecg samples in {self.split} dataset.")
        print(f"There are [{self.pos_cnt}] death samples and [{self.neg_cnt}] alive samples, positive ratio is {self.pos_cnt/len(self.ecg_idx_list):.4f}.")

    def load_ecg(self, ecg_path):

        ecg = scipy.io.loadmat(ecg_path)
        curr_sample_rate = ecg['curr_sample_rate']
        curr_sample_size = ecg['curr_sample_size']

        if (curr_sample_rate != self.sample_rate):
            raise Exception(f"ERROR: current sample rate: {curr_sample_rate}, need: {self.sample_rate}, ecg path: {ecg_path}")
        
        if (curr_sample_size != self.target_size):
            raise Exception(f"ERROR: current sample size: {curr_sample_size}, need: {self.target_size}, ecg path: {ecg_path}")

        feats = torch.from_numpy(ecg['feats'])
        feats = feats.float()
        return feats

    def __getitem__(self, index):

        ecg_idx = self.ecg_idx_list[index]
        segment_0_path = self.ecg_data[ecg_idx]["segment_0_path"]
        segment_1_path = self.ecg_data[ecg_idx]["segment_1_path"]

        res = {'ecg_idx': ecg_idx}
        res["segment_0"] = self.load_ecg(segment_0_path)
        res["segment_1"] = self.load_ecg(segment_1_path)
        res["label"] = torch.tensor(self.ecg_data[ecg_idx]["label"], dtype=torch.float32)
        return res

    def __len__(self):
        return len(self.ecg_idx_list)
    

# 死亡窗口后的标签置为无效
def adjust_labels(labels):
    death_idx = np.argmax(labels) if sum(labels) > 0 else len(labels)
    if death_idx != len(labels):
        labels[death_idx+1:] = [-1] * len(labels[death_idx+1:])
    return labels


class ECGSurvivalDataset(BaseDataset):
    def __init__(
        self,
        split: str="train",
        records_path: str=None,
        data_dir: str=None,
        sample_rate = 500,
        target_size = 2500,
        time_bins = [0, 6, 12, 24, 48, 72, 168, 336, 672],
        **kwargs,
    ):
        super().__init__()

        self.split = split
        self.sample_rate = sample_rate
        self.target_size = target_size
        
        if data_dir is not None:
            self.data_dir = data_dir 
        elif os.path.isdir(os.path.join(os.path.dirname(records_path), 'segmented')):
            self.data_dir = os.path.join(os.path.dirname(records_path), 'segmented')
        else:
            raise Exception(f"ERROR: cannot find a valid data_dir !")

        self.ecg_data = {}
        self.pos_cnt = 0
        self.neg_cnt = 0

        with open(records_path, "r") as f:
            # 读取行标题
            f.readline().strip()
            for _, line in enumerate(f):
                items = line.strip().split(",", maxsplit=7)
                assert len(items) == 8, f"ERROR: There are something wrong when read records at [{line}], length is {len(items)}"

                # 是所需的数据集划分:
                if items[2] == self.split:
                    ecg_path = os.path.join(self.data_dir, items[1])
                    label_list = ast.literal_eval(items[7].strip('"'))

                    duration_idx = bisect.bisect_right(time_bins, float(items[6]))-1
                    if duration_idx > len(time_bins) - 2:
                        duration_idx = len(time_bins) - 2

                    event = int(items[5]) if float(items[6]) < time_bins[-1] else 0
                    
                    self.ecg_data[items[0]] = {
                        "segment_0_path": ecg_path.replace('.mat', '_0.mat'),
                        "segment_1_path": ecg_path.replace('.mat', '_1.mat'),
                        "label": adjust_labels(label_list),
                        "duration_idx": duration_idx,
                        'event': event,
                        'delta_hours': float(items[6])
                    }
                    if event == 1:
                        self.pos_cnt += 1
                    elif event == 0:
                        self.neg_cnt += 1
                    else:
                        raise Exception(f"ERROR: There is a unvalid label at {items[1]}, the label is {label_list}!")

        self.ecg_idx_list = list(self.ecg_data.keys())
        print(f"Loaded [{len(self.ecg_idx_list)}] ecg samples in {self.split} dataset.")
        print(f"There are [{self.pos_cnt}] death samples and [{self.neg_cnt}] alive samples, positive ratio is {self.pos_cnt/len(self.ecg_idx_list):.4f}.")

    def load_ecg(self, ecg_path):

        if not os.path.exists(ecg_path):
            raise Exception(f"ERROR: Cannot find {ecg_path}!")

        ecg = scipy.io.loadmat(ecg_path)
        curr_sample_rate = ecg['curr_sample_rate']
        curr_sample_size = ecg['curr_sample_size']

        if (curr_sample_rate != self.sample_rate):
            raise Exception(f"ERROR: current sample rate: {curr_sample_rate}, need: {self.sample_rate}, ecg path: {ecg_path}")
        
        if (curr_sample_size != self.target_size):
            raise Exception(f"ERROR: current sample size: {curr_sample_size}, need: {self.target_size}, ecg path: {ecg_path}")

        feats = torch.from_numpy(ecg['feats'])
        feats = feats.float()
        return feats

    def __getitem__(self, index):

        ecg_idx = self.ecg_idx_list[index]
        segment_0_path = self.ecg_data[ecg_idx]["segment_0_path"]
        segment_1_path = self.ecg_data[ecg_idx]["segment_1_path"]

        res = {'ecg_idx': ecg_idx}
        res["segment_0"] = self.load_ecg(segment_0_path)
        res["segment_1"] = self.load_ecg(segment_1_path)
        res["label"] = torch.LongTensor(self.ecg_data[ecg_idx]["label"])
        res['duration_idx'] = torch.tensor(self.ecg_data[ecg_idx]["duration_idx"])
        res['event'] = torch.tensor(self.ecg_data[ecg_idx]["event"], dtype=torch.float32)
        res['delta_hours'] = torch.tensor(self.ecg_data[ecg_idx]["delta_hours"], dtype=torch.float32)
        return res

    def __len__(self):
        return len(self.ecg_idx_list)
    

class ECGSurvivalPycoxDataset(BaseDataset):
    def __init__(
        self,
        split: str="train",
        records_path: str=None,
        data_dir: str=None,
        sample_rate = 500,
        target_size = 2500,
        **kwargs,
    ):
        super().__init__()

        self.split = split
        self.sample_rate = sample_rate
        self.target_size = target_size
        
        if data_dir is not None:
            self.data_dir = data_dir 
        elif os.path.isdir(os.path.join(os.path.dirname(records_path), 'segmented')):
            self.data_dir = os.path.join(os.path.dirname(records_path), 'segmented')
        else:
            raise Exception(f"ERROR: cannot find a valid data_dir !")

        self.ecg_data = {}
        self.pos_cnt = 0
        self.neg_cnt = 0

        with open(records_path, "r") as f:
            # 读取行标题
            f.readline().strip()
            for _, line in enumerate(f):
                items = line.strip().split(",", maxsplit=7)
                assert len(items) == 8, f"ERROR: There are something wrong when read records at [{line}], length is {len(items)}"

                # 是所需的数据集划分:
                if items[2] == self.split:
                    ecg_path = os.path.join(self.data_dir, items[1])

                    duration_idx = bisect.bisect_right(time_bins, float(items[6]))-1
                    if duration_idx > len(time_bins) - 2:
                        duration_idx = len(time_bins) - 2
                    
                    self.ecg_data[items[0]] = {
                        "segment_0_path": ecg_path.replace('.mat', '_0.mat'),
                        "segment_1_path": ecg_path.replace('.mat', '_1.mat'),
                        "durations": duration_idx,
                        'events': int(items[5]),
                    }
                    if 1 == int(items[5]):
                        self.pos_cnt += 1
                    elif 0 == int(items[5]):
                        self.neg_cnt += 1
                    else:
                        raise Exception(f"ERROR: There is a unvalid label at {items[1]}, the label is {items[7]}!")

        self.ecg_idx_list = list(self.ecg_data.keys())
        print(f"Loaded [{len(self.ecg_idx_list)}] ecg samples in {self.split} dataset.")
        print(f"There are [{self.pos_cnt}] death samples and [{self.neg_cnt}] alive samples, positive ratio is {self.pos_cnt/len(self.ecg_idx_list):.4f}.")

    def load_ecg(self, ecg_path):

        if not os.path.exists(ecg_path):
            raise Exception(f"ERROR: Cannot find {ecg_path}!")

        ecg = scipy.io.loadmat(ecg_path)
        curr_sample_rate = ecg['curr_sample_rate']
        curr_sample_size = ecg['curr_sample_size']

        if (curr_sample_rate != self.sample_rate):
            raise Exception(f"ERROR: current sample rate: {curr_sample_rate}, need: {self.sample_rate}, ecg path: {ecg_path}")
        
        if (curr_sample_size != self.target_size):
            raise Exception(f"ERROR: current sample size: {curr_sample_size}, need: {self.target_size}, ecg path: {ecg_path}")

        feats = torch.from_numpy(ecg['feats'])
        feats = feats.float()
        return feats

    def __getitem__(self, index):

        ecg_idx = self.ecg_idx_list[index]
        segment_0_path = self.ecg_data[ecg_idx]["segment_0_path"]
        segment_1_path = self.ecg_data[ecg_idx]["segment_1_path"]

        res = {'ecg_idx': ecg_idx}
        res["segment_0"] = self.load_ecg(segment_0_path)
        res["segment_1"] = self.load_ecg(segment_1_path)
        res['durations'] = torch.tensor(self.ecg_data[ecg_idx]["durations"])
        res['events'] = torch.tensor(self.ecg_data[ecg_idx]["events"])
        return res

    def __len__(self):
        return len(self.ecg_idx_list)
    

class SurvivalPredictDataset(BaseDataset):
    def __init__(
        self,
        split: str="train",
        records_path: str=None,
        data_dir: str=None,
        ecg_sample_rate = 500,
        ecg_target_size = 2500,
        text_tokenizer = None,
        text_max_length = 512,
        text_max_segment = 3,
        time_bins = [0, 6, 12, 24, 48, 72, 168, 336, 672],
        **kwargs,
    ):
        super().__init__()

        self.split = split
        self.ecg_sample_rate = ecg_sample_rate
        self.ecg_target_size = ecg_target_size

        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length
        self.text_max_segment = text_max_segment

        if data_dir is not None:
            self.data_dir = data_dir 
        elif os.path.isdir(os.path.join(os.path.dirname(records_path), 'segmented')):
            self.data_dir = os.path.join(os.path.dirname(records_path), 'segmented')
        else:
            raise Exception(f"ERROR: cannot find a valid data_dir !")

        self.survival_data = {}
        self.pos_cnt = 0
        self.neg_cnt = 0

        print("time_bin: ", time_bins)

        meta_data = pd.read_csv(records_path)
        for i in tqdm(range(len(meta_data))):
            items = meta_data.iloc[i].values
            assert len(items) == 25, f"ERROR: There are something wrong when read records at line [{i}], length is {len(items)}"

            # 是所需的数据集划分:
            if str(items[2]) == self.split:
                ecg_path = os.path.join(self.data_dir, str(items[1]))

                duration_idx = bisect.bisect_right(time_bins, float(items[6]))-1
                if duration_idx > len(time_bins) - 2:
                    duration_idx = len(time_bins) - 2

                # 在规定时间前没死的，统一认为活着，在规定时间前死的，按照死亡时间算（在家死还是医院死由筛数据时决定）
                event = int(items[5]) if float(items[6]) < time_bins[-1] else 0

                # ====================================== 处理文本数据 ==========================================================

                text = str(items[7]).replace('\n', ' ')
                full_text = f"predict mortality: {text}"
                texts = self.preprocess_text(full_text)
                text_segments = self.split_text(texts)
                
                # 对文本信息进行预先的编码 
                text_encodings = {
                    "input_ids": [],
                    "attention_mask": []
                }

                for segment in text_segments:
                    encoded = self.text_tokenizer(
                        segment,
                        max_length=self.text_max_length,
                        truncation="longest_first",
                        padding="max_length",
                        return_tensors="pt",
                        add_special_tokens=True
                    )
                    text_encodings["input_ids"].append(encoded["input_ids"].squeeze(0))  # 移除batch维
                    text_encodings["attention_mask"].append(encoded["attention_mask"].squeeze(0))

                # 处理不足max_segments的情况（填充空白段落）
                while len(text_encodings["input_ids"]) < self.text_max_segment:
                    text_encodings["input_ids"].append(torch.zeros(self.text_max_length, dtype=torch.long))
                    text_encodings["attention_mask"].append(torch.zeros(self.text_max_length, dtype=torch.long))


                # ======================================== 处理tabnet数据 ========================================================

                tabnet_data = items[8:]
                tabnet_data[0] = float(tabnet_data[0]) / 100.0
                
                self.survival_data[int(items[0])] = {
                    "segment_0_path": ecg_path.replace('.mat', '_0.mat'),
                    "segment_1_path": ecg_path.replace('.mat', '_1.mat'),
                    "duration_idx": duration_idx,
                    'event': event,
                    'delta_hours': float(items[6]),
                    "input_ids": torch.stack(text_encodings["input_ids"]),  # [text_max_segments, text_max_length]
                    "attention_mask": torch.stack(text_encodings["attention_mask"]),
                    "tabnet_data": torch.tensor(tabnet_data.astype(np.float32))
                }

                # 统计信息
                if event == 1:
                    self.pos_cnt += 1
                elif event == 0:
                    self.neg_cnt += 1
                else:
                    raise Exception(f"ERROR: There is a unvalid label at {items[1]}, the event is {event}, duration_id is {duration_idx}!")

        self.data_idx_list = list(self.survival_data.keys())
        print(f"Loaded [{len(self.data_idx_list)}] samples in {self.split} dataset.")
        print(f"There are [{self.pos_cnt}] death samples and [{self.neg_cnt}] alive samples, positive ratio is {self.pos_cnt/len(self.data_idx_list):.4f}.")

    def load_ecg(self, ecg_path):

        if not os.path.exists(ecg_path):
            raise Exception(f"ERROR: Cannot find {ecg_path}!")

        ecg = scipy.io.loadmat(ecg_path)
        curr_sample_rate = ecg['curr_sample_rate']
        curr_sample_size = ecg['curr_sample_size']

        if (curr_sample_rate != self.ecg_sample_rate):
            raise Exception(f"ERROR: current sample rate: {curr_sample_rate}, need: {self.ecg_sample_rate}, ecg path: {ecg_path}")
        
        if (curr_sample_size != self.ecg_target_size):
            raise Exception(f"ERROR: current sample size: {curr_sample_size}, need: {self.ecg_target_size}, ecg path: {ecg_path}")

        feats = torch.from_numpy(ecg['feats'])
        feats = feats.float()
        return feats

    def preprocess_text(self, text):
        # 替换日期（YYYY-MM-DD 格式）
        text = re.sub(r"\d{4}-\d{2}-\d{2}", "[DATE]", text)
        # 替换数字（整数或小数）
        text = re.sub(r"\b\d+\.?\d*\b", "[NUM]", text)
        # 替换患者相关词（不区分大小写）
        text = re.sub(r"\b(patient|pt|subject)\b", "[PT]", text, flags=re.IGNORECASE)
        return text

    def split_text(self, text):
        # 整个分成3段，每段取前512个token
        segment_length = len(text) // self.text_max_segment
        sentences = [text[i:i + segment_length] for i in range(0, len(text), segment_length)]
        return sentences[:self.text_max_segment]  # 截断到最大段数  

    def __getitem__(self, index):

        data_idx = self.data_idx_list[index]
        segment_0_path = self.survival_data[data_idx]["segment_0_path"]
        segment_1_path = self.survival_data[data_idx]["segment_1_path"]

        res = {'data_idx': data_idx}
        res["segment_0"] = self.load_ecg(segment_0_path)
        res["segment_1"] = self.load_ecg(segment_1_path)
        res['duration_idx'] = torch.tensor(self.survival_data[data_idx]["duration_idx"])
        res['event'] = torch.tensor(self.survival_data[data_idx]["event"], dtype=torch.float32)
        res['delta_hours'] = torch.tensor(self.survival_data[data_idx]["delta_hours"], dtype=torch.float32)
        res['input_ids'] = self.survival_data[data_idx]["input_ids"]
        res['attention_mask'] = self.survival_data[data_idx]["attention_mask"]
        res['tabnet_data'] = self.survival_data[data_idx]["tabnet_data"]
        return res

    def __len__(self):
        return len(self.data_idx_list)