import itertools
from typing import Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm


def split_dataset(indices: Sequence[int], labels:Sequence[Sequence[str]], num_test: int, num_dev: int) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """
    Generates train-dev-test split over the indices.
    This dataset is multi-label, therefore an approximate iterative solution  is used to split the data.
    See: http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html#module-skmultilearn.model_selection.iterative_stratification

    :param num_dev: size of dev set
    :param num_test: size of test set
    :param indices: used by the `Dataloader` instance
    :param labels: Label sets that correspond to each index
    :return: Tuple containing the training, development and testing splits, correspondingly
    """

    # Fix random state for reproducibility
    rng = np.random.default_rng(512)

    # Make the inputs numpy arrays
    X = np.reshape(np.asarray(indices), (-1, 1))

    labels = [frozenset(l) for l in labels]

    voc = {w:ix for ix, w in enumerate(set(labels))}

    y = np.asarray([voc[l] for l in labels])

    proportion_test = num_test / len(indices)

    # Do train - test split
    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=proportion_test, random_state=rng.integers(0, 1000))

    proportion_dev = num_dev / len(X_train)

    # Split further train into train - dev
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=proportion_dev, random_state=rng.integers(0, 1000))


    return X_train.reshape((-1, )).tolist(), X_dev.reshape((-1, )).tolist(), X_test.reshape((-1, )).tolist()