""" Trainer script for REACH BERT """

# Create data module
# Create model instance
# Create and configure trainer

# Configure checkpoint saving
# Run validation benchmarks

from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from data_loaders import ReachDataset
from data_loaders.reach_data_module import ReachDataModule
from data_loaders.callbacks import ShiftMasksCallback
from modeling.reach_bert import ReachBert


def main(hparams):
    # Load the data module and query the parameters
    dataset = ReachDataset(data_dir = "/media/evo870/data/reach-bert-data/bert_files_w_negatives",
                           masked_data_dir= "/media/evo870/data/reach-bert-data/masked_bert_files_w_negatives", overwrite_index=False,
                           debug = True)

    # Load the tokenizer model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    # This is a hyper param tuned using data_processing/sequence_length_stats.py
    max_seq_len = 69

    data_module = ReachDataModule(dataset = dataset, tokenizer = tokenizer, max_seq_len = max_seq_len, batch_size=80, num_workers=8)

    model = ReachBert(num_interactions=dataset.num_interactions, num_tags=dataset.num_tags)
    checkpoint_callback = ModelCheckpoint(dirpath='ckpts', save_top_k=3, auto_insert_metric_name=False, monitor="Loss/Combined Val")
    masked_change_callback = ShiftMasksCallback()
    trainer = Trainer(gpus= 1, val_check_interval=1., callbacks=[checkpoint_callback, masked_change_callback]) # Do validation every 10% of an epoch
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()

    main(args)