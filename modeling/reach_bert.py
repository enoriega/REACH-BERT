""" Implement REACH BERT model """
from abc import ABC, ABCMeta
from typing import Mapping, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn, Tensor
from torch.nn import functional as F, CrossEntropyLoss, MSELoss, BCELoss
from torchmetrics import MetricCollection, Precision, Recall, F1Score
from transformers import AutoModel, AutoModelForSequenceClassification

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
            F1Score(num_classes= num_interactions, average = 'micro', multiclass=False),
            Precision(num_classes= num_interactions, average = 'macro', multiclass=False),
            Recall(num_classes= num_interactions, average = 'macro', multiclass=False)])

        tag_metrics = MetricCollection([
            F1Score(num_classes=num_tags, average='macro'),
            Precision(num_classes=num_tags, average='macro'),
            Recall(num_classes=num_tags, average='macro')])

        self.val_label_metrics = label_metrics.clone(prefix='val_')
        self.test_label_metrics = label_metrics.clone(prefix='test_')
        ##############

        # Load BERT ckpt, the foundation to our model
        self.transformer = AutoModel.from_pretrained(backbone_model_name, num_labels=num_interactions)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_interactions)

    def forward(self, **inputs:Mapping[str, Tensor]):
        """ Forward pass of our model """

        # Pass the tensor through the transformer
        outputs = self.transformer(**inputs) # We have to unpack the arguments of the transformer's fwd method

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        ret = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return ret

    # region Shared methods
    def __step(self, batch: ReachBertInput, batch_idx: int) -> Optional[STEP_OUTPUT]:
        """ Body of the train/val/test step """

        encoding, tags, labels = batch.encoding, batch.tags, batch.labels

        outputs = self(**encoding.data)

        logits =  outputs[0]

        if labels is not None:
            if self.num_interactions == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # interaction_weights = torch.tensor(self.interaction_weights,
                #                                    device=self.device) if self.interaction_weights else None
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_interactions), labels)
            outputs = (loss,) + outputs


        label_loss, logits = outputs


        # tag_weights = torch.tensor(self.tag_weights, device=self.device) if self.tag_weights else None

        label_predictions = (logits >= 0.5).int()


        # Return all data
        return {
            "loss": label_loss, #+ tag_loss,
            "label_loss": label_loss.detach(),
            "label_targets":  labels.int(),
            "label_predictions": label_predictions.int(),
            # "tag_loss": tag_loss.detach(),
            # "tag_targets": tag_targets.detach(),
            # "tag_predictions": tag_predictions.detach()
        }

    def __log(self,
              step_kind:str,
              data:dict,
              label_metrics:Optional[MetricCollection] = None,
              tag_metrics:Optional[MetricCollection] = None):
        """ Generic log function for the metrics for the current step """

        def _log_collection(task_name:str, metrics:MetricCollection):
            """ Helper function for logging a metrics collection"""
            self.log(f"{task_name}/F1 {step_kind}", metrics.F1Score, on_step=False, on_epoch=True)
            self.log(f"{task_name}/Precision {step_kind}", metrics.Precision, on_step=False, on_epoch=True)
            self.log(f"{task_name}/Recall {step_kind}", metrics.Recall, on_step=False, on_epoch=True)

        if label_metrics and tag_metrics:
            # Compute the metrics from the predictions
            label_metrics(data["label_predictions"], data["label_targets"])
            tag_metrics(data["tag_predictions"], data["tag_targets"])

            # Log metrics for both tasks
            _log_collection("Label", label_metrics)
            # _log_collection("Tag", tag_metrics)

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
        data = self.__step(batch, batch_idx)

        # Log losses
        # self.log(f"Train Loss/Label", data['label_loss'], on_step=True)
        # self.log(f"Train Loss/Tag", data['tag_loss'], on_step=True)
        self.log(f"Train Loss", data['loss'], on_step=True)

        return data


    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        data = self.__step(batch, batch_idx)

        # Log
        self.log(f"Val Loss", data['loss'], on_epoch=True)

        self.val_label_metrics(data['label_predictions'], data['label_targets'])

        self.log("Val F1", self.test_label_metrics.F1Score, on_step=False, on_epoch=True)
        self.log("Val Precision", self.test_label_metrics.F1Score, on_step=False, on_epoch=True)
        self.log("Val Recall", self.test_label_metrics.F1Score, on_step=False, on_epoch=True)

        return data

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        data = self.__step(batch, batch_idx)

        self.test_label_metrics(data['label_predictions'], data['label_targets'])

        self.log("Test F1", self.test_label_metrics.F1Score, on_step=False, on_epoch=True)
        self.log("Test Precision", self.test_label_metrics.Precision, on_step=False, on_epoch=True)
        self.log("Test Recall", self.test_label_metrics.Recall, on_step=False, on_epoch=True)

        return data

    # def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     preds,  targets = list(), list()
    #
    #     for o in outputs:
    #         preds.append(o['label_predictions'])
    #         targets.append(o['label_targets'])
    #
    #     preds = torch.cat(preds, dim=0).cpu().numpy()
    #     targets = torch.cat(targets, dim=0).cpu().numpy()
    #
    #     np.save('preds_x.npy', preds)
    #     np.save('targetx.npy', targets)
    # endregion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
        return optimizer
