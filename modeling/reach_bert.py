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

        self.num_interactions = num_interactions
        self.num_tags = num_tags

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
        data = self._step(batch, batch_idx)

        return data['label_loss'] + data['tag_loss']

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        data = self._step(batch, batch_idx)

        return data['label_loss'] + data['tag_loss']

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        data = self._step(batch, batch_idx)

        return data['label_loss'] + data['tag_loss']

    def _step(self, batch: ReachBertInput, batch_idx: int) -> Optional[STEP_OUTPUT]:

        inputs, tags, labels = batch.features, batch.tags, batch.labels
        x = self(**inputs)

        # Pass the relevant outputs of the transformers to the appropriate heads
        pooler_tensor = x['last_hidden_state'][:, 0, :]  # This is the [CLS] embedding
        pooler_output = self._pooler_head(pooler_tensor)

        label_loss = F.binary_cross_entropy_with_logits(pooler_output, labels)

        # Compute the predictions
        label_predictions = (torch.sigmoid(pooler_output) >= 0.5).float().detach()

        tags_tensor = x['last_hidden_state']
        tags_outputs = self._tagger_head(tags_tensor)

        # Build a boolean mask to compute the loss only of the attended tokens (ignore the padded entries of the batch)
        tags_mask = inputs['attention_mask'].flatten().bool()

        # Flatten the batch to measure a two-dimension sequence of logits, then select only the attended logits
        tags_outputs = tags_outputs.reshape((-1, self.num_tags))[tags_mask, :]

        # Do the same for the tag targets
        tags = tags.reshape((-1, self.num_tags))[tags_mask, :]

        # Get the predictions and targets
        tag_predictions = torch.max(tags_outputs, 1)[1].float()
        tag_targets = torch.max(tags, 1)[1].float()

        tag_loss  = F.cross_entropy(tags_outputs, tags)


        # Return all data
        return {
            "label_loss": label_loss,
            "label_targets":  labels,
            "label_predictions": label_predictions,
            "tag_loss": tag_loss,
            "tag_targets": tag_targets.detach(),
            "tag_predictions": tag_predictions.detach()
        }



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer