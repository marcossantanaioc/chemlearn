import copy
import logging
from typing import Dict
import dill as pickle
import numpy as np
from sklearn.base import is_classifier
from chemlearn.data.dataset import MolDataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ChemSklearnModel:
    def __init__(self, model):
        self.model = model

    def fit(self, x, y):
        """
        Trains the model using training set(x, y).
        Parameters
        ----------
        x
            Training features
        y
            Training target
        """
        self.model.fit(x, y)
        return self

    def predict(self, x):
        """
        Gather predictions from a fitted model.
        """
        if is_classifier(self.model):
            return self.model.predict_proba(x)
        else:
            return self.model.predict(x)

    def set_params(self, params: Dict):

        """
        Set the parameters of this model.

        The method works on simple models as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            model parameters.

        Returns
        -------
        self : model instance
            model instance.
        """

        if hasattr(self.model, 'named_steps'):
            self.model.steps[-1][1].set_params(**params)

        elif 'Chain' in self.model.__class__.__name__ and hasattr(self.model, 'base_model'):
            self.model.base_model.set_params(**params)
        else:
            self.model.set_params(**params)

        return self

    def copy(self):
        """Makes a copy of `self`. """
        return copy.deepcopy(self)

    def export(self, filename: str = 'moldataset.pkl'):
        """Save a model to a file name or opened file"""

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(filename: str):
        """Load a pickle file from a file name or opened file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def clean(self, attr):
        """Removes an attribute."""
        del type(self).attr


class ChemLearner:
    """
    Create a ChemModel and implement methods for training and validating the model

    Attributes
    ----------
        dataset
            A `MolDataset`.
        model
            An instance of ChemSKModel. train_data A PandasDataset for the training set. valid_data A PandasDataset
            for the validation set. featurizer A `MolFeaturizer` object. data A dictionary where each key is a column in
            `df`, and values are the column's values. columns List of columns in `df` dtype Type of `target_variable`.
            See [sklearn documentation for details](
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html) job_type
            Whether the dataset is for a regression or classification task, based on `dtype`.
        """

    def __init__(self, model, dataset: MolDataset):

        self.model = ChemSklearnModel(model)
        self.dataset = dataset
        self.target_variable = getattr(dataset, 'target_variable')
        self.train_data = getattr(dataset, 'train_data')
        self.valid_data = getattr(dataset, 'valid_data')
        self.dtype = getattr(dataset, 'dtype')
        self.job_type = getattr(dataset, 'job_type')
        self.featurizer = getattr(dataset, 'featurizer')
        self.c = getattr(dataset, 'c', None)
        self.classes = getattr(dataset, 'classes', None)

    def get_data(self, data: MolDataset, return_target: bool = True):
        x = np.stack(data.data['features'])
        if return_target:
            y = np.stack(data.data[self.target_variable])
            return x, y
        return x

    def fit(self, params: Dict = {}):
        x, y = self.get_data(self.train_data)
        if params:
            self.model.set_params(params)
        self.model.fit(x, y)
        return self

    def predict(self, data: MolDataset):
        x = self.get_data(data, return_target=False)
        return self.model.predict(x)

