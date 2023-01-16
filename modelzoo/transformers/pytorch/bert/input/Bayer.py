import os
from functools import partial
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

class CustomDataset(Dataset):
    def __init__(self, data_path, cols=None):
        self.df = pd.read_parquet(data_path).reset_index(drop=True)
        self.cols = cols
        if self.cols is not None:
            assert all(_col in self.df.columns for _col in self.cols)
            self.df = self.df[cols]
        print("\ninitialized CustomDataset with df", self.df)
        # sequence_aa  chain  cdr1_start  cdr2_start  cdr3_start  cdr1_length  cdr2_length  cdr3_length  sequence_length
        # sequence_aa is unspaced chars
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.df.iloc[[idx]].to_dict('records')[0]

def BERTCollate_fn(data, tokenizer=None, max_sequence_length=None, pad_to_max_sequence_length=False, mlm_probability=0.15, return_input_seqs=False):
    # batching function to benefit from multiple workers in the dataloader
    # performs tokenization on the fly
    # TODO: use additional metadata cols e.g. chain and cdrX_start/cdrX_length -> token_type_ids or non-uniform masking / loss scaling
    with torch.inference_mode():
        minibatch = tokenizer([row["sequence_aa"].replace("", " ")[1:-1] for row in data], return_special_tokens_mask=True)
        # TODO: handle CODON vocab of character triplets
        minibatch = tokenizer.pad(minibatch, return_tensors="pt", pad_to_multiple_of=1)
        if return_input_seqs:
            minibatch["input_seqs"] = [row["sequence_aa"] for row in data]
        bs = minibatch["input_ids"].shape[0]
        padded_length = minibatch["input_ids"].shape[1]
        minibatch["input_ids"] = minibatch["input_ids"].int()
        minibatch["attention_mask"] = minibatch["attention_mask"].int()
        probability_matrix = torch.full(minibatch["input_ids"].shape, mlm_probability)
        probability_matrix.masked_fill_(minibatch["special_tokens_mask"], value=0.0)
        # TODO: implement variable masking_p based on e.g. FWs/CDRs
        masked_indices = torch.zeros_like(probability_matrix)
        while torch.sum(masked_indices) == 0:
            # we ensure there is a least 1 masked token to avoid inf
            masked_indices = torch.bernoulli(probability_matrix).bool()
        max_predictions_per_seq = torch.max(torch.sum(masked_indices, dim=1))
        minibatch["masked_lm_positions"] = torch.zeros((bs, max_predictions_per_seq), dtype=torch.int32)
        # TODO: is there a way to vectorize this operation without for loop ?
        for _i in range(bs):
            minibatch["masked_lm_positions"][_i, :torch.sum(masked_indices[_i, :])] = torch.nonzero(masked_indices[_i, :], as_tuple=True)[0]
        minibatch["masked_lm_weights"] = torch.zeros((bs, max_predictions_per_seq), dtype=torch.float32).fill_(bs/torch.sum(masked_indices))
        # TODO: should the value of masked_lm_weights be set by the config mlm_loss_weight (constant)
        minibatch["labels"] = minibatch["input_ids"].clone()
        minibatch["labels"][~masked_indices] = -100
        # hardcode 80% mask, 10% keep, 10% swap on minibatch["input_ids"]
        indices_replaced = torch.bernoulli(torch.full(minibatch["labels"].shape, 0.8)).bool() & masked_indices
        minibatch["input_ids"][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(minibatch["labels"].shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), minibatch["labels"].shape, dtype=torch.int32)
        minibatch["input_ids"][indices_random] = random_words[indices_random]
        if pad_to_max_sequence_length and padded_length < max_sequence_length:
            minibatch["input_ids"] = torch.cat([minibatch["input_ids"], torch.zeros((bs, max_sequence_length-padded_length), dtype=torch.int32)], dim=1)
            minibatch["attention_mask"] = torch.cat([minibatch["attention_mask"], torch.zeros((bs, max_sequence_length-padded_length), dtype=torch.int32)], dim=1)
            minibatch["labels"] = torch.cat([minibatch["labels"], torch.zeros((bs, max_sequence_length-padded_length), dtype=torch.int32).fill_(-100)], dim=1)
        # we correctly get sequences with BOS/EOS, padded with 0 and attention mask with 0 on padded positions
        minibatch.pop('special_tokens_mask', None)
        minibatch.pop('token_type_ids', None)
        minibatch["masked_lm_mask"] = minibatch["attention_mask"].clone()
        # TODO: check the needed masked_lm_mask key https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/pytorch/bert/model.py#L192
    return dict(minibatch)

class BayerDataset():
    def __init__(self, params) -> None:
        self.data_dir = params["data_dir"]
        self.vocab_file = params["vocab_file"]
        self.batch_size = params["batch_size"]
        self.max_sequence_length = params["max_sequence_length"]
        self.pad_to_max_sequence_length = params.get("pad_to_max_sequence_length", True)
        self.masked_lm_prob = params.get("masked_lm_prob", 0.15)
        self.return_input_seqs = params.get("return_input_seqs", False)
        self.num_workers = params.get("num_workers", 0)
        self.persistent_workers = params.get("persistent_workers", False)

    def create_dataloader(self, is_training):
        cols = ["sequence_aa", "chain", "sequence_length"]+[f"cdr{i}_start" for i in range(1, 4)]+[f"cdr{i}_length" for i in range(1, 4)]
        dataset = CustomDataset(self.data_dir, cols=cols)
        tokenizer = BertTokenizer(vocab_file=self.vocab_file, do_lower_case=False)

        collate_fn = partial(BERTCollate_fn, 
                             tokenizer=tokenizer, 
                             max_sequence_length=self.max_sequence_length, 
                             pad_to_max_sequence_length=self.pad_to_max_sequence_length,
                             mlm_probability=self.masked_lm_prob, 
                             return_input_seqs=self.return_input_seqs)

        dataloader = DataLoader(dataset, 
                                batch_size=self.batch_size, 
                                shuffle=False, 
                                num_workers=0, 
                                collate_fn=collate_fn
                                )
        return dataloader
