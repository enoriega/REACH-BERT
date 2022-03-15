""" Implement REACH BERT model """
from typing import Mapping, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from data_loaders.reach_data_module import ReachBertInput


class ReachBert(pl.LightningModule):
    """ REACH BERT """

    def __init__(self, num_interactions: int, num_tags: int):
        super(ReachBert, self).__init__()

        # Load BERT ckpt, the foundation to our model
        self.transformer = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

        # TODO: Do research to find out the STOA arch for each of the heads
        # This is the pooler head, which will be used to predict the multilabel task of predicting the interactions
        # present in the current input
        self._pooler_head = nn.Sequential(
            #nn.Linear(768, 768),           # I assume that using the hidden states as bare features will backpropagate better for fine-tunning. Have to verify.
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(768, num_interactions),
        )

        # The tagger head is responsible for predicting the tags of the individual tokens (Participant, Trigger or None)
        self._tagger_head = nn.Sequential(
            #nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(768, num_tags),
        )

    def forward(self, **inputs:Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        # Pass the tensor through the transformer
        return self.transformer(**inputs) # We have to unpack the arguments of the transformer's fwd method

    def training_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch: ReachBertInput, batch_idx: int) -> Optional[STEP_OUTPUT]:

        inputs, tags, labels = batch.features, batch.tags, batch.labels
        x = self(**inputs)

        # Pass the relevant outputs of the transformers to the appropriate heads
        pooler_tensor = x['last_hidden_state'][:, 0, :]  # This is the [CLS] embedding
        pooler_output = self._pooler_head(pooler_tensor)

        label_loss = F.binary_cross_entropy_with_logits(pooler_output, labels)

        tags_tensor = x['last_hidden_state']
        tags_outputs = self._tagger_head(tags_tensor)

        tags_loss  = F.binary_cross_entropy_with_logits(tags_outputs, tags, reduction='none') # Don't reduce because we are going to select only the elements of the attended tokens

        # Use the attention mask to choose the tokens that where attended
        tags_loss = tags_loss[inputs['attention_mask'], :].flatten().mean()

        # Add both losses
        return label_loss + tags_loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return torch.tensor([0.])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer