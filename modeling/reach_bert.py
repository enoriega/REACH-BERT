""" Implement REACH BERT model """
from abc import ABC, ABCMeta
from typing import Mapping, Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
from torch.nn import functional as F
from torchmetrics import MetricCollection, Precision, Recall, F1Score
from transformers import AutoModel

from data_loaders.reach_data_module import ReachBertInput


class ReachBert(pl.LightningModule, metaclass=ABCMeta):
    """ REACH BERT """

    def __init__(self,
                 backbone_model_name: str,
                 num_interactions: int,
                 num_tags: int,
                 interaction_weights:Optional[Sequence[float]] = None,
                 tag_weights:Optional[Sequence[float]] = None):

        super(ReachBert, self).__init__()

        # Bookkeeping
        self.num_interactions = num_interactions
        self.num_tags = num_tags
        self.interaction_weights = interaction_weights
        self.tag_weights = tag_weights

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
        """ Forward pass of our model """

        # Pass the tensor through the transformer
        return self.transformer(**inputs) # We have to unpack the arguments of the transformer's fwd method

    # region Shared methods
    def __step(self, batch: ReachBertInput, batch_idx: int) -> Optional[STEP_OUTPUT]:
        """ Body of the train/val/test step """

        inputs, tags, labels = batch.features, batch.tags, batch.labels
        x = self(**inputs)

        interaction_weights = torch.tensor(self.interaction_weights, device=self.device) if self.interaction_weights else None
        tag_weights = torch.tensor(self.tag_weights, device=self.device) if self.tag_weights else None

        # Pass the relevant outputs of the transformers to the appropriate heads
        pooler_tensor = x['last_hidden_state'][:, 0, :]  # This is the [CLS] embedding
        pooler_output = self._pooler_head(pooler_tensor)

        label_loss = F.binary_cross_entropy_with_logits(pooler_output, labels, weight=interaction_weights)

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

        tag_loss  = F.cross_entropy(tags_outputs, tags, weight=tag_weights)

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

    def __step_end(self, batch_parts, step_kind:str, label_metrics:MetricCollection, tag_metrics:MetricCollection) -> STEP_OUTPUT:
        """ Generic step end hook """

        # Merge together all the batch parts, in case it runs in multiple GPUs
        data = self.__merge_batch_parts(batch_parts)

        # Do logging
        self.__log(step_kind, data, label_metrics, tag_metrics)

        # Return the merged step's data
        return data

    def __log(self, step_kind:str, data:dict, label_metrics:MetricCollection, tag_metrics:MetricCollection):
        """ Generic log function for the metrics for the current step """

        def _log_collection(task_name:str, metrics:MetricCollection):
            """ Helper function for logging a metrics collection"""
            self.log(f"{task_name}/F1 {step_kind}", metrics.F1Score, on_step=False, on_epoch=True)
            self.log(f"{task_name}/Precision {step_kind}", metrics.Precision, on_step=False, on_epoch=True)
            self.log(f"{task_name}/Recall {step_kind}", metrics.Recall, on_step=False, on_epoch=True)

        # Compute the metrics from the predictions
        label_metrics(data["label_predictions"], data["label_targets"])
        tag_metrics(data["tag_predictions"], data["tag_targets"])

        # Log metrics for both tasks
        _log_collection("Label", label_metrics)
        _log_collection("Tag", tag_metrics)

        # Log losses
        self.log(f"Loss/Label {step_kind}", data['label_loss'], on_step=True)
        self.log(f"Loss/Tag {step_kind}", data['tag_loss'], on_step=True)
        self.log(f"Loss/Combined {step_kind}", data['loss'], on_step=True)

    def __merge_batch_parts(self, batch_parts):
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
    # endregion

    # region Step hooks
    def training_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.__step(batch, batch_idx)

    def training_step_end(self, batch_parts):
        return self.__step_end(batch_parts, "Train", self.train_label_metrics, self.train_tag_metrics)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.__step(batch, batch_idx)

    def validation_step_end(self, batch_parts) -> STEP_OUTPUT:
        return self.__step_end(batch_parts, "Val", self.val_label_metrics, self.val_tag_metrics)
    # endregion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
