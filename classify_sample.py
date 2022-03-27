""" Classifies a sample of the test set for a manual error analysis """
import itertools
import random
from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd
import torch.cuda
from transformers import PreTrainedTokenizer, AutoTokenizer

from data_loaders import ReachDataset, DatasetIndex
from data_loaders.reach_data_module import collator
from data_loaders.utils import collapse_labels
from modeling.reach_bert import ReachBert
from numpy import random

def main(ckpt_path, dataset_path):
    # Load the data

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = ReachDataset(data_dir=dataset_path, data_hook=collapse_labels)
    model = ReachBert.load_from_checkpoint(ckpt_path, backbone_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_interactions=dataset.num_interactions, num_tags=dataset.num_tags).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    # Sample a batch
    rng = random.default_rng(1991)
    indices = rng.integers(low=0, high=len(dataset), size=1000)

    batch = [dataset[ix] for ix in indices]

    positives, negatives = [], []
    for b in batch:
        if len(b.event_labels) >  0:
            positives.append(b)
        else:
            negatives.append(b)

    batch = positives[:50] + negatives[:50]

    classify(model, batch, dataset.index, tokenizer)

def classify(model, batch, index: DatasetIndex, tokenizer:PreTrainedTokenizer):

    ret = []

    for item in batch:
        sent = " ".join(item.words)
        labels = item.event_labels
        ret.append({
            "sent": sent,
            "labels": labels
        })



    collated = collator(batch, max_seq_len=69, ds_index=index, tokenizer=tokenizer, train=False)

    encoding, tags, labels = collated.encoding, collated.tags, collated.labels


    encoding.to(model.device)

    output = model(**encoding)

    logits = output[0]

    preds = (logits >= .5).int()

    # Map one-hot to indices
    indices = [tuple(row) for row in (preds == 1).nonzero().cpu().numpy()]

    pred_counter = Counter()
    gt_counter = Counter()

    for item_ix, indices in itertools.groupby(indices, key=lambda r:r[0]):
        pred_labels = [index.codes_interaction[r[1]] for r in indices]
        ret[item_ix]['pred_labels'] = pred_labels
        pred_counter.update(pred_labels)
        gt_counter.update(ret[item_ix]['labels'])

    for key in pred_counter:
        pred_counter[key] /= len(batch)

    for key in gt_counter:
        gt_counter[key] /= len(batch)

    # pprint(pred_counter)
    # print()
    # pprint(gt_counter)

    pc = pd.DataFrame.from_dict(pred_counter, orient="index", columns=['predicted'])
    gtc = pd.DataFrame.from_dict(gt_counter, orient="index", columns=['gt'])

    frame = pc.join(gtc, how='outer').fillna(0).sort_values('predicted', ascending=False)

    # unions, intersections = set(), set()
    # for r in ret:
    #     unions |= set(r['labels'])
    #     intersections &= set(r['pred_labels'])

    # js = list()
    # for a, b in itertools.product(ret, ret):
    #     a = set(a['pred_labels'])
    #     b = set(b['pred_labels'])
    #
    #     j = len(a & b) / len(a | b)
    #     js.append(j)
    #
    #
    # jaccard = np.asarray(js)
    # print(f'Avg Jaccard: {jaccard.mean()} ({jaccard.std()})')

    for r in ret:
        pprint(r)
        print()



if __name__ == "__main__":

    main(
        "ckpts/reach_bert-epoch_2-step_39917-val_loss_0.133.ckpt",
        "/media/evo870/data/reach-bert-data/bert_files_w_negatives"
    )