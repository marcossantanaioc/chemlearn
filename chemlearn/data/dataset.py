import copy
import datetime
from typing import Union
import dill as pickle
import pandas as pd
from cheminftools.tools.featurizer import MolFeaturizer
from sklearn.utils.multiclass import type_of_target, unique_labels
from chemlearn.utils.splitters import Splitter

TARGET2TYPE = {'continuous': 'regression',
               'binary': 'classification',
               'continuous-multioutput': 'regression-multi',
               'multiclass': 'multiclass',
               'multilabel-indicator': 'multilabel'}

TYPE2TARGET = {v: k for k, v in TARGET2TYPE.items()}


class PandasDataset:
    """
    Create a dataset from a pandas dataframe.
    Attributes
    ----------
    df
        Input dataframe.
    smiles_column
        Column in `df` with SMILES.
    target_variable
        Column in `df` with prediction target.
    featurizer
        A `MolFeaturizer` object.
    data
        A dictionary where each key is a column in `df`,
        and values are the column's values.
    columns
        List of columns in `df`
    dtype
        Type of `target_variable`.
        See [sklearn documentation for details](https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html)
    job_type
        Whether the dataset is for a regression or classification task,
        based on `dtype`.
    """
    def __init__(self, df: pd.DataFrame, smiles_column: str, target_variable: str, featurizer: MolFeaturizer):
        self.df = df
        self.featurizer = featurizer
        self.data = dict(zip(df.columns, df.to_numpy().T))
        self.smiles_column = smiles_column
        self.columns = df.columns.tolist()
        self.target_variable = target_variable
        self.dtype = type_of_target(self.df[target_variable])
        self.job_type = TARGET2TYPE.get(self.dtype, None)

    @property
    def classes(self):
        """
        Returns the classes in `self.data` if
        `self.job_type` is a classification task.
        Returns
        -------
        labels
            Classes present in `self.target_variable`

        """
        if self.job_type in ['classification', 'multiclass']:
            return unique_labels(self.df[self.target_variable])

    @property
    def c(self):
        """
        Returns the number of classes in `self.data` if
        `self.job_type` is a classification task.
        Returns
        -------
        number_of_classes
            Number of classes present in `self.target_variable`

        """
        if self.job_type in ['classification', 'multiclass']:
            return len(self.classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idxs):
        res = {key: self.data[key][idxs] for key in self.data.keys()}
        return res

    def create_dataset(self):
        fp = self.featurizer.transform(self.data[self.smiles_column]).tolist()
        _df = pd.DataFrame(self.data)
        _df['features'] = fp
        self.data = dict(zip(_df.columns, _df.to_numpy().T))
        return self


class MolDataset(PandasDataset):
    """
    Creates a dataset ready for modeling with molecular data.
    Attributes
    ----------
    df
        Input dataframe.
    smiles_column
        Column in `df` with SMILES.
    target_variable
        Column in `df` with prediction target.
    featurizer
        A `MolFeaturizer` object.
    data
        A dictionary where each key is a column in `df`,
        and values are the column's values.
    columns
        List of columns in `df`
    dtype
        Type of `target_variable`.
        See [sklearn documentation for details](https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html)
    job_type
        Whether the dataset is for a regression or classification task,
        based on `dtype`.
    """

    def __init__(self, data_path: Union, smiles_column: str, target_variable: str, featurizer: MolFeaturizer,
                 splitter: Splitter):
        super().__init__(pd.read_csv(data_path), smiles_column=smiles_column, target_variable=target_variable,
                         featurizer=featurizer)

        self.dataset = self.create_dataset()
        self.data_path = data_path
        self.splits = splitter.split(self.df)

    @property
    def all(self):
        return self.dataset

    @property
    def train_idx(self):
        return self.splits['train_idx']

    @property
    def valid_idx(self):
        return self.splits['valid_idx']

    @property
    def train_data(self):
        dset = pd.DataFrame(self.dataset[self.train_idx])
        return PandasDataset(dset, self.smiles_column, self.target_variable, self.featurizer)

    @property
    def valid_data(self):
        dset = pd.DataFrame(self.dataset[self.valid_idx])
        return PandasDataset(dset, self.smiles_column, self.target_variable, self.featurizer)

    @property
    def timestamp(self):
        now = datetime.datetime.now()
        return now.strftime("%d-%m-%Y")

    def copy(self):
        """
        Creates a copy of `self`
        """
        return copy.deepcopy(self)

    def __str__(self):
        return f'Time stamp: {self.timestamp}\nTarget: {self.target_variable}\nData type: {self.dtype}\nNumber of ' \
               f'compounds: {len(self.df)}\nThe first entry is: {self[0]}'

    def export(self, filename: str = 'moldataset.pkl'):
        """Save the dict dataset to a file name or opened file"""

        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    @staticmethod
    def load_pickle(filename: str):
        """Load a pickle file from a file name or opened file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
