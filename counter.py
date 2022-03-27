from data_loaders.reach_dataset import ReachDataset
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method

from data_loaders.utils import collapse_labels

if __name__ == '__main__':

    ds = ReachDataset('/media/evo870/data/reach-bert-data/bert_files_w_negatives', overwrite_index=False, data_hook=collapse_labels)
    ts = ds.train_dataset()

    def count_labels(indices, ds):
        c = Counter()
        for ix in indices:
            try:
                i = ds[ix]
                labels = i.event_labels
                if "positive-activation" in labels or "negative-activation" in labels:
                    c['activation'] += 1
                elif len(labels) == 0:
                    c['empty'] += 1
                else:
                    c['other'] += 1
            except:
                x = 0
        return c

    pool = ProcessPoolExecutor()

    x = list()
    for indices in range(16):
        l = len(ds.index.train_indices) // 16
        ixs = ds.index.train_indices[indices * l:(indices + 1) * l]
        x.append(ixs)

    futures = list()

    print("Submitting")
    for  indices in x:
        futures.append(pool.submit(count_labels, indices, ts))

    print("Done submitting")
    res = Counter()
    for f in tqdm(as_completed(futures)):
        res += f.result()

    print(res)

