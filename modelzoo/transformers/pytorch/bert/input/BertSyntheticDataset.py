"""
Processors for synthetic data for BERT
"""
import math

import numpy as np
import torch
from torch.utils.data import Dataset


class BertDataset(Dataset):
    """
    A class representing a BertDataset inheriting torch.utils.data.Dataset.
    """

    def __init__(self, data, data_processor):
        self.data = data
        self.batch_size = data_processor.batch_size
        self.length = data_processor.num_examples
        self.disable_nsp = data_processor.disable_nsp
        self.gather_mlm_labels = data_processor.gather_mlm_labels
        self.dynamic_mlm_scale = data_processor.dynamic_mlm_scale
        super(BertDataset, self).__init__()

    def __getitem__(self, index):
        feature = {
            "input_ids": self.data["input_ids"][index],
            "attention_mask": self.data["attention_mask"][index],
            "token_type_ids": self.data["token_type_ids"][index],
            "labels": self.data["labels"][index],
            "masked_lm_mask": self.data["masked_lm_mask"][index],
        }
        if not self.disable_nsp:
            feature["next_sentence_label"] = self.data["next_sentence_labels"][
                index
            ]
        if self.gather_mlm_labels:
            feature["masked_lm_positions"] = self.data["masked_lm_positions"][
                index
            ]
        if self.dynamic_mlm_scale:
            # Not technically collated over the batch, but its the right
            # type/shape, and close to the right meaning
            scale = float(self.batch_size) / np.sum(feature["masked_lm_mask"])
            feature["mlm_loss_scale"] = scale.reshape(1).astype(np.float16)
        return feature

    def __len__(self):
        return self.length


class BertSyntheticDataProcessor:
    """
    Synthetic dataset generator.

    :param dict params: List of training input parameters for creating dataset
    """

    def __init__(self, params):
        self.vocab_size = params["vocab_size"]
        self.num_examples = params["num_examples"]
        self.max_seq_len = params["max_sequence_length"]
        self.gather_mlm_labels = params.get("gather_mlm_labels", True)
        if self.gather_mlm_labels:
            self.max_pred = params.get(
                "max_predictions_per_seq", math.ceil(0.15 * self.max_seq_len)
            )
        else:
            self.max_pred = self.max_seq_len

        self.dynamic_mlm_scale = params.get("dynamic_mlm_scale", False)
        self.batch_size = params["batch_size"]
        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", 1)
        self.sampler = params.get("sampler", None)
        self.batch_sampler = params.get("batch_sampler", None)
        self.num_workers = params.get("num_workers", 0)
        self.pin_memory = params.get("pin_memory", False)
        self.drop_last = params.get("drop_last", True)
        self.timeout = params.get("timeout", 0)
        self.mp_type = (
            torch.float16 if params.get("mixed_precision") else torch.float32
        )
        self.disable_nsp = params.get("disable_nsp", False)

    def create_dataloader(self, is_training=True):
        """
        Create dataloader.

        :returns: dataloader
        """

        np.random.seed(self.shuffle_seed)
        data = dict()

        # Input mask
        attention_mask = np.ones(
            (self.num_examples, self.max_seq_len), dtype=np.int32
        )
        seq_mid_idx = np.cast["int32"](self.max_seq_len / 2)
        for i in range(self.num_examples):
            start_idx = np.random.randint(seq_mid_idx, self.max_seq_len + 1)
            attention_mask[i, start_idx : self.max_seq_len] = 0
        data["attention_mask"] = attention_mask

        # Input tokens
        data["input_ids"] = (
            np.random.randint(
                low=0,
                high=self.vocab_size,
                size=(self.num_examples, self.max_seq_len),
                dtype=np.int32,
            )
            * attention_mask
        )  # 0 out padded positions

        # Token type ids
        token_type_ids = np.zeros(
            shape=(self.num_examples, self.max_seq_len), dtype=np.int32
        )
        token_type_ids[:, seq_mid_idx : self.max_seq_len] = 1
        data["token_type_ids"] = token_type_ids * attention_mask

        # Mask for MLM predictions
        masked_lm_mask = np.ones(
            shape=(self.num_examples, self.max_pred), dtype=np.int32
        )
        pred_mid_idx = np.cast["int32"](self.max_pred / 2)
        for i in range(self.num_examples):
            start_idx = np.random.randint(pred_mid_idx, self.max_pred + 1)
            masked_lm_mask[i, start_idx : self.max_pred] = 0

        data["masked_lm_mask"] = masked_lm_mask

        if self.gather_mlm_labels:
            # Masked token positions
            mlm_positions = np.zeros(
                shape=(self.num_examples, self.max_pred), dtype=np.int32
            )
            for i in range(0, self.num_examples):
                num_tokens = np.count_nonzero(data["attention_mask"][i])
                index_values = np.arange(int(num_tokens))
                np.random.shuffle(index_values)
                index = index_values[0 : self.max_pred]
                index.sort()
                # Ensure masked_lm_positions is padded to max_pred.
                index_shape = index.shape[0]
                index = np.pad(
                    index, (0, self.max_pred - index_shape), 'constant'
                )
                mlm_positions[i] = index

            # MLM labels
            data["masked_lm_positions"] = mlm_positions * masked_lm_mask

            labels = np.zeros(
                shape=(self.num_examples, self.max_pred), dtype=np.int32
            )
            for i in range(0, self.num_examples):
                for j in range(0, self.max_pred):
                    pos = int(data["masked_lm_positions"][i, j])
                    labels[i, j] = data["input_ids"][i, pos]

            # Pad unused labels with -100
            labels = labels * masked_lm_mask
            labels = labels - 100 * (1 - masked_lm_mask)

        else:
            labels = (
                data["attention_mask"] * data["input_ids"]
                + (1 - data["attention_mask"]) * -100
            )

        data["labels"] = labels

        # NSP labels
        if not self.disable_nsp:
            data["next_sentence_labels"] = np.random.randint(
                low=0, high=2, size=self.num_examples, dtype=np.int32
            )

        return torch.utils.data.DataLoader(
            BertDataset(data, self),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
        )