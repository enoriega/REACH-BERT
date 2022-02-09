""" Implement REACH BERT model """
from typing import Mapping

import pytorch_lightning as pl
from torch import nn, Tensor
from transformers import AutoModel


class REACHBert(pl.LightningModule):
    """ REACH BERT """

    def __init__(self, num_interations: int, num_tags: int):
        super(REACHBert, self).__init__()

        # Load BERT ckpt, the foundation to our model
        self.transformer = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

        # TODO: Do research to find out the STOA arch for each of the heads
        # This is the pooler head, which will be used to predict the multilabel task of predicting the interactions
        # present in the current input
        self._pooler_head = nn.Sequential(
            #nn.Linear(768, 768),           # I assume that using the hidden states as bare features will backpropagate better for fine-tunning. Have to verify.
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(768, num_interations),
        )

        # The tagger head is responsible for predicting the tags of the individual tokens (Participant, Trigger or None)
        self._tagger_head = nn.Sequential(
            #nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(768, num_tags),
        )

    def forward(self, inputs:Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        # Pass the tensor through the transformer
        x = self.transformer(**inputs) # We have to unpack the arguments of the transformer's fwd method

        # Pass the relevant outputs of the transformers to the appropriate heads
        # pooler_tensor = x['pooler_input'] # This is the output of a FF with tanh activation layer on top of the [CLS] token
        pooler_tensor = x['last_hidden_state'][:, 0, :] # This is the [CLS] embedding
        pooler_output = self._pooler_head(pooler_tensor)

        tags_tensor = x['last_hidden_state'][:, 1:-1, :] # Discard the first token ([CLS]) and the last token ([SEP])
        tags_outputs = self._tagger_head(tags_tensor)

        # Return the tensor outputs
        return {"interactions_logits": pooler_output,  "tags_logits": tags_outputs}

