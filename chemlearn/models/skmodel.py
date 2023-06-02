import copy
import logging
from typing import Dict, Union, Tuple, Any, Optional
from numpy.typing import ArrayLike
import dill as pickle
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from tqdm import tqdm
from chemlearn.data.dataset import MolDataset, PandasDataset
from chemlearn.utils.splitters import CrossValidationSplitter, Splitter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
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

    def predict(self, x) -> ArrayLike:
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

    def __init__(self, model, dataset: MolDataset, metric):

        self.model = ChemSklearnModel(model)
        self.dataset = dataset
        self.metric = metric
        self.metric_name = metric.metric.__name__
        self.target_variable = getattr(dataset, 'target_variable')
        self.train_data = getattr(dataset, 'train_data')
        self.valid_data = getattr(dataset, 'valid_data')
        self.dtype = getattr(dataset, 'dtype')
        self.job_type = getattr(dataset, 'job_type')
        self.featurizer = getattr(dataset, 'featurizer')
        self.c = getattr(dataset, 'c', None)
        self.classes = getattr(dataset, 'classes', None)

    def get_data(self, data: Union[MolDataset, PandasDataset, Dict], return_target: bool = True) -> Tuple[Any, Any]:
        if isinstance(data, (MolDataset, PandasDataset)):
            data = data.data
        x = np.stack(data['features'])
        if return_target:
            return x, np.stack(data[self.target_variable])
        return x  # type: ignore

    def fit(self, x, y, params: Optional[Dict[Any, Any]] = None):
        if not params:
            params = {}
        self.model.set_params(params)
        self.model.fit(x, y)
        return self

    def predict(self, data: Union[MolDataset, PandasDataset, Dict]) -> ArrayLike:
        x = self.get_data(data, return_target=False)
        return self.model.predict(x)

    def cross_val(self,
                  splitter: Splitter,
                  n_splits: int = 5,
                  params: Optional[Dict[Any, Any]] = None) -> pd.DataFrame:

        """
        Performs k-fold cross-validation.

        Parameters
        ----------
        splitter
            An iterator to generate cross-validation folds.
        n_splits
            The number of cross-validation folds
        params
            Hyperparameters to pass to the model.

        Returns
        -------
        cv_results
            Summary of predictive performance.
        """
        cv_iterator = CrossValidationSplitter(splitter=splitter, n_splits=n_splits)

        if params:
            self.model.set_params(params)

        gather_metrics = {self.metric_name: [], 'Fold': []}  # type: ignore

        logger.info('Performing cross-validation\nParameters:')
        logger.info(f'splits = {n_splits}')
        logger.info(f'iterator = {splitter.__class__.__name__}')

        for fold, (train_split, valid_split) in tqdm(enumerate(cv_iterator.split(self.train_data)), total=n_splits,
                                                     position=0, leave=False):
            xtrain, ytrain = self.get_data(self.train_data[train_split])

            logger.info(f'Fitting on fold {fold}')
            logger.info(f'Training on {len(train_split)} samples.')
            logger.info(f'Validating on {len(valid_split)} samples.')

            self.fit(x=xtrain, y=ytrain)

            logger.info(f'Finished fold {fold}')

            preds = self.predict(self.train_data[valid_split])
            yvalid = self.train_data[valid_split][self.target_variable]

            score = self.metric.score(yvalid, preds)
            gather_metrics[self.metric_name].append(score)
            gather_metrics['Fold'].append(fold)

        cv_results = pd.DataFrame(gather_metrics)

        logger.debug(cv_results)
        logger.debug(f'##################################################################')
        logger.info(f'##################################################################')

        return cv_results

# if __name__ == "__main__":  # type: ignore
#     from sklearn.ensemble import RandomForestRegressor
#     from chemlearn.utils.splitters import TrainTestSplitter
#     from cheminftools.tools.featurizer import MolFeaturizer
#     from chemlearn.models.validation.metrics import MSEScore
#
#     m = RandomForestRegressor()
#     splitter = TrainTestSplitter()
#     featurizer = MolFeaturizer('morgan')
#     moldataset = MolDataset(data_path='/home/marcossantana/PycharmProjects/cheminftools/data/data.csv',
#                             smiles_column='smiles', target_variable='pIC50', featurizer=featurizer, splitter=splitter)
#     metric = MSEScore(squared=False)
#
#     chemmodel = ChemLearner(m, dataset=moldataset, metric=metric)
#     cv_results = chemmodel.cross_val(splitter=splitter, n_splits=5)
#     print(cv_results)
