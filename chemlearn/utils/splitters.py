import numbers
from collections import defaultdict
from typing import Collection, Union, List, Dict, Set
import numpy as np
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles


class Splitter:
    def __init__(self, test_size: float = 0.2, random_state: int = None, **kwargs):
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.random_state = random_state
        for key, value in kwargs.items():
            setattr(self, key, value)

    def split(self, data):
        pass


class TrainTestSplitter(Splitter):
    """Split `items` into random train and test subsets using sklearn train_test_split utility."""

    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = None,
                 stratify: np.ndarray = None,
                 shuffle: bool = True):
        super().__init__(test_size=test_size, random_state=random_state, stratify=stratify, shuffle=shuffle)

    def split(self, data):
        from sklearn.model_selection import train_test_split
        train, valid = train_test_split(list(range(len(data))), **self.__dict__)
        return {'train_idx': train, 'valid_idx': valid}


class CrossValidationSplitter:
    """A Splitter to convert splitters into iterators"""

    def __init__(self, splitter, n_splits: int = 10, random_state: int = None):
        self.splitter = splitter
        self.n_splits = n_splits

    def split(self, data):
        """Returns a split generator."""
        k = 0
        # random = check_random_state(self.random_state)

        while k < self.n_splits:
            folds = self.splitter.split(data)
            train_idx, valid_idx = folds['train_idx'], folds['valid_idx']
            yield train_idx, valid_idx
            k += 1


def check_random_state(seed):
    import numbers
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class RandomSplitter(Splitter):
    """Create function that splits `items` between train/val with `valid_pct` randomly."""

    def __init__(self, test_size: float = 0.2, random_state: float = None):
        super().__init__(test_size=test_size, random_state=random_state)

    def split(self, data):
        import torch
        if self.random_state is not None and isinstance(self.random_state, numbers.Integral):
            torch.manual_seed(self.random_state)

        rand_idx = list(torch.randperm(len(data)).numpy())
        cut = int(self.test_size * len(data))

        return {'train_idx': rand_idx[cut:], 'valid_idx': rand_idx[:cut]}


class IndexSplitter(Splitter):
    """Split `items` so that `val_idx` are in the validation set and the others in the training set"""

    def __init__(self, test_idx: Collection, random_state: float = None):
        super().__init__(test_size=0, random_state=random_state)
        self.test_idx = test_idx

    def split(self, data):
        train_idx = np.setdiff1d(np.array(list(range(len(data)))), np.array(self.test_idx))
        # return L(train_idx, use_list=True), L(self.valid_idx, use_list=True)
        return {'train_idx': train_idx, 'valid_idx': self.test_idx}


def scaffold_to_smiles(smiles: List[str],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    Parameters
    ----------
    smiles
        A list of SMILES or RDKit molecules.
    use_indices
        Whether to map to the SMILES's index in :code:`mols` rather than
        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    Returns
    -------
    A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, smi in enumerate(smiles):
        scaffold = MurckoScaffoldSmilesFromSmiles(smi)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smi)

    return scaffolds


class ScaffoldSplitter(Splitter):
    """
    Splits a :class:`~MolDataset` by scaffold so that no molecules sharing a scaffold are in different splits.
    Parameters
    ----------
    test_size
        Percentage of data to use as a test set.
    smiles_column
        Column where to find SMILES.
    n_splits
        Number of splits to generate.
    balanced
        Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    random_state
        Random seed for shuffling when doing balanced splitting.
    Returns
    -------
    A tuple of train and valid indices
    """

    def __init__(self,
                 test_size: float = 0.2,
                 smiles_column: str = 'smiles',
                 n_splits: int = 10,
                 balanced: bool = True,
                 random_state: int = 0):
        super().__init__(test_size=test_size, random_state=random_state, smiles_column=smiles_column)

        self.sizes = (1 - test_size, test_size)
        self.balanced = balanced
        assert sum(self.sizes) == 1
        self.n_splits = n_splits

    def split(self, data):

        if not hasattr(data, 'smiles'):
            raise AttributeError(f'Attribute smiles is missing from input. Please provide a valid MolDataset.')

        smiles = data.__getattribute__('smiles')
        random_state = check_random_state(self.random_state)

        # Split
        train_size, val_size = self.sizes[0] * len(data), self.sizes[1] * len(data)
        train, val = [], []
        train_scaffold_count, val_scaffold_count = 0, 0

        # Map from scaffold to index in the data
        scaffold_to_indices = scaffold_to_smiles(smiles, use_indices=True)

        if self.balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
            index_sets = list(scaffold_to_indices.values())
            big_index_sets = []
            small_index_sets = []
            for index_set in index_sets:
                if len(index_set) > val_size / 2:
                    big_index_sets.append(index_set)
                else:
                    small_index_sets.append(index_set)

            index_sets = big_index_sets + small_index_sets
        else:  # Sort from largest to smallest scaffold sets
            index_sets = sorted(list(scaffold_to_indices.values()),
                                key=lambda index_set: len(index_set),
                                reverse=True)

        index_sets = random_state.permutation(index_sets)

        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1

        return {'train_idx': train, 'valid_idx': val}

# scaffold_splitter = ScaffoldSplitter(test_size=0.2, balanced=True)
# print(scaffold_splitter.balanced)
