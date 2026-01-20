import os
import random
import torch

from data.amazon import AmazonReviews
from data.schemas import SeqBatch, TokenizedSeqBatch
from enum import Enum
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"
import pickle

class RecDataset(Enum):
    AMAZON = 1
    ML_1M = 2
    ML_32M = 3


DATASET_NAME_TO_RAW_DATASET = {
    RecDataset.AMAZON: AmazonReviews,
    # RecDataset.ML_1M: RawMovieLens1M,
    # RecDataset.ML_32M: RawMovieLens32M,
}


DATASET_NAME_TO_MAX_SEQ_LEN = {
    RecDataset.AMAZON: 20,
    # RecDataset.ML_1M: 200,
    # RecDataset.ML_32M: 200
}


class ItemData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        train_test_split: str = "all",
        **kwargs
    ) -> None:
        
        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, **kwargs)
        
        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:

            raw_data.process(max_seq_len=max_seq_len)
        
        if train_test_split == "train":
            filt = raw_data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_data.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(raw_data.data["item"]["x"][:,0], dtype=bool)

        # print(raw_data.data['item']['x'].shape, filt.shape)
        self.item_data, self.item_text = raw_data.data["item"]["x"][filt], raw_data.data["item"]["text"][filt]

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :]
        # print(x.shape)
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            x_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=torch.ones_like(item_ids, dtype=bool),
            label=torch.tensor([])
            # ids_neg=torch.tensor([-1])
        )


class SeqData(Dataset):
    def __init__(
        self,
        root: str,
        is_train: bool = True,
        subsample: bool = False,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        save_tokens=False,
        *args,
        **kwargs
    ) -> None:
        assert (not subsample) or is_train, "Can only subsample on training split."

        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, **kwargs)

        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)

        split = "train" if is_train else "test"
        self.subsample = subsample
        self.sequence_data = raw_data.data[("user", "rated", "item")]["history"][split]
        self.save_tokens = save_tokens
        if not self.subsample:
            self.sequence_data["itemId"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(l[-max_seq_len:]) for l in self.sequence_data["itemId"]] + [torch.zeros(max_seq_len)],
                batch_first=True,
                padding_value=-1,
                padding_side='left'
            )[:-1]

        self._max_seq_len = max_seq_len
        self.item_data = raw_data.data["item"]["x"]
        self.split = split

        self.num_items = self.item_data.shape[0]
    
    @property
    def max_seq_len(self):
        return self._max_seq_len

    def __len__(self):
        return self.sequence_data["userId"].shape[0]
  
    def __getitem__(self, idx):
        user_ids = self.sequence_data["userId"][idx]
        if self.subsample:
            seq = self.sequence_data["itemId"][idx] + self.sequence_data["itemId_fut"][idx].tolist()
            start_idx = random.randint(0, max(0, len(seq)-3))
            end_idx = random.randint(start_idx+3, start_idx+self.max_seq_len+1)
            sample = seq[start_idx: end_idx]

            item_ids = torch.tensor([-1] * (self.max_seq_len - len(sample[:-1])) + sample[:-1])
            # y o n
            # item_ids_fut = torch.tensor([-1] * (self.max_seq_len - len(sample[1:])) + sample[1:])
            item_ids_fut = torch.tensor([-1] * (self.max_seq_len - len(sample[1:])) + sample[1:])
            label = sample[-1]
        else:
            item_ids = self.sequence_data["itemId"][idx]
            if self.save_tokens:
                item_ids = torch.tensor(item_ids)
            item_ids_fut = self.sequence_data["itemId_fut"][idx]
            label = item_ids_fut

            item_ids_neg = torch.tensor([-1])
        
        assert (item_ids >= -1).all(), "Invalid movie id found"
        x = self.item_data[item_ids, :]
        x[item_ids == -1] = -1

        x_fut = self.item_data[item_ids_fut, :]
        x_fut[item_ids_fut == -1] = -1

        # print(f"========== idx {idx} ============")
        # print(item_ids.shape)
        # print(item_ids_fut.shape)
        # print(item_ids_neg.shape)

        return SeqBatch(
            user_ids=user_ids,
            ids=item_ids,
            ids_fut=item_ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=(item_ids >= 0),
            label=label
        )

if __name__ == "__main__":
    dataset = SeqData(
        "/DATA/DATANAS2/zhangyip/tencent_proj/proj/RQ-VAE-Recommender/dataset/amazon_qwen", 
        dataset=RecDataset.AMAZON, 
        split="beauty", 
        force_process=False
    )
    d = dataset[0]
    print(d)
