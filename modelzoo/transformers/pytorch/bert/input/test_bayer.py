# This file is used to test the Bayer dataset

from Bayer import BayerDataset, BayerAltDataset


params = {
    "data_dir": "/testshare/vinay/bmlm-bayer-cerebras/sample_data/OAS_allspecies_filtered_clustered095.csv",
    "vocab_file": "/testshare/vinay/bmlm-bayer-cerebras/sample_data/bert_vocab.txt",
    "batch_size": 2,
    "max_sequence_length": 160,
    "pad_to_max_sequence_length": True,
    "masked_lm_prob": 0.15,
    "return_input_seqs": False,
    "num_workers": 1,
    "persistent_workers": True
}


# BayerDataset
dataset = BayerDataset(params)
dataloader = dataset.create_dataloader(is_training=True)
for i, batch in enumerate(dataloader):
    print(batch)
    break


# BayerAltDataset
dataset = BayerAltDataset(params)
dataloader = dataset.create_dataloader(is_training=True)
for i, batch in enumerate(dataloader):
    print(batch)
    break
