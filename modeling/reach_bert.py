""" Implement REACH BERT model """
from typing import Mapping, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
from torch.nn import functional as F
from torchmetrics import MetricCollection, Precision, Recall, F1Score
from transformers import AutoModel

from data_loaders.reach_data_module import ReachBertInput


class ReachBert(pl.LightningModule):
    """ REACH BERT """

    def __init__(self, backbone_model_name: str, num_interactions: int, num_tags: int):
        super(ReachBert, self).__init__()

        # Bookkeeping
        self.num_interactions = num_interactions
        self.num_tags = num_tags

        # Metrics
        label_metrics = MetricCollection([
            F1Score(num_classes= num_interactions, average = 'macro', mdmc_average='samplewise'),
            Precision(num_classes= num_interactions, average = 'macro', mdmc_average='samplewise'),
            Recall(num_classes= num_interactions, average = 'macro', mdmc_average='samplewise')])

        tag_metrics = MetricCollection([
            F1Score(num_classes=num_tags, average='macro'),
            Precision(num_classes=num_tags, average='macro'),
            Recall(num_classes=num_tags, average='macro')])

        self.train_label_metrics = label_metrics.clone(prefix='train_')
        self.val_label_metrics = label_metrics.clone(prefix='val_')

        self.train_tag_metrics = tag_metrics.clone(prefix='train_')
        self.val_tag_metrics = tag_metrics.clone(prefix='val_')
        ##############

        # Load BERT ckpt, the foundation to our model
        self.transformer = AutoModel.from_pretrained(backbone_model_name)

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
        return self._step(batch, batch_idx)

    def training_step_end(self, batch_parts):

        data = self._merge_batch_parts(batch_parts)

        self.train_label_metrics(data["label_predictions"], data["label_targets"])
        self.train_tag_metrics(data["tag_predictions"], data["tag_targets"])

        self.log("Label/F1 Train", self.train_label_metrics.F1Score, on_step=False, on_epoch=True)
        self.log("Label/Precision Train", self.train_label_metrics.Precision, on_step=False, on_epoch=True)
        self.log("Label/Recall Train", self.train_label_metrics.Recall, on_step=False, on_epoch=True)

        self.log("Tag/F1 Train", self.train_tag_metrics.F1Score, on_step=False, on_epoch=True)
        self.log("Tag/Precision Train", self.train_tag_metrics.Precision, on_step=False, on_epoch=True)
        self.log("Tag/Recall Train", self.train_tag_metrics.Recall, on_step=False, on_epoch=True)

        self.log("Loss/Label Training", data['label_loss'], on_step=True)
        self.log("Loss/Tag Training", data['tag_loss'], on_step=True)
        self.log("Loss/Combined Training", data['loss'], on_step=True)

        return data


    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._step(batch, batch_idx)

    def validation_step_end(self, batch_parts) -> STEP_OUTPUT:
        data = self._merge_batch_parts(batch_parts)

        self.val_label_metrics(data["label_predictions"], data["label_targets"])
        self.val_tag_metrics(data["tag_predictions"], data["tag_targets"])

        self.log("Label/F1 Val", self.val_label_metrics.F1Score, on_step=False, on_epoch=True)
        self.log("Label/Precision Val", self.val_label_metrics.Precision, on_step=False, on_epoch=True)
        self.log("Label/Recall Val", self.val_label_metrics.Recall, on_step=False, on_epoch=True)

        self.log("Tag/F1 Val", self.val_tag_metrics.F1Score, on_step=False, on_epoch=True)
        self.log("Tag/Precision Val", self.val_tag_metrics.Precision, on_step=False, on_epoch=True)
        self.log("Tag/Recall Val", self.val_tag_metrics.Recall, on_step=False, on_epoch=True)

        self.log("Loss/Label Val", data['label_loss'], on_step=False, on_epoch=True)
        self.log("Loss/Tag Val", data['tag_loss'], on_step=False, on_epoch=True)
        self.log("Loss/Combined Val", data['loss'], on_step=False, on_epoch=True)

        return data

    # def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        # data = self._step(batch, batch_idx)
        #
        # return data['label_loss'] + data['tag_loss']

    # def test_step_end(self, batch_parts) -> STEP_OUTPUT:
    #     return self._step_end(batch_parts)

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
        tag_predictions = torch.max(tags_outputs, 1)[1]
        tag_targets = torch.max(tags, 1)[1]

        tag_loss  = F.cross_entropy(tags_outputs, tags)

        # Return all data
        return {
            "loss": label_loss + tag_loss,
            "label_loss": label_loss.detach(),
            "label_targets":  labels.int(),
            "label_predictions": label_predictions.int(),
            "tag_loss": tag_loss.detach(),
            "tag_targets": tag_targets.detach(),
            "tag_predictions": tag_predictions.detach()
        }

    def _merge_batch_parts(self, batch_parts):
        """ Merge batch parts when using data parallel computation (i.e. on the HPC) """

        if type(batch_parts) == dict:
            merged_data = dict()
            for key in batch_parts:
                if key.endswith("loss"):
                    merged_data[key] = batch_parts[key].mean()
                elif key.startswith("label"):
                    merged_data[key] = batch_parts[key].reshape((-1, self.num_interactions))
                else:
                    merged_data[key] = batch_parts[key].flatten()

        else:
            merged_data = batch_parts

        return merged_data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
