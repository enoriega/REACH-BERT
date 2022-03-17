""" Trainer script for REACH BERT """

# Create and configure trainer
# Configure checkpoint saving
# Run validation benchmarks

from pytorch_lightning.utilities.cli import LightningCLI

from data_loaders.reach_data_module import ReachDataModule
from modeling.reach_bert import ReachBert


class ReachBertCLI(LightningCLI):
    """ Custom CLI class to wire parameters from the dataset to the model """

    def add_arguments_to_parser(self, parser):
        # These arguments are precomputed on the index and don't need to be specified in config
        parser.link_arguments("data.num_interactions", "model.num_interactions", apply_on="instantiate")
        parser.link_arguments("data.num_tags", "model.num_tags", apply_on="instantiate")


if __name__ == "__main__":
    ReachBertCLI(ReachBert, ReachDataModule)