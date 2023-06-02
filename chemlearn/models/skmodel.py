import copy
import logging
from typing import Dict, Union, Tuple, Any, Optional
from numpy.typing import ArrayLike
import dill as pickle
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tqdm import tqdm
from chemlearn.data.dataset import MolDataset, PandasDataset
from chemlearn.utils.splitters import CrossValidationSplitter, Splitter
from sklearn.metrics import mean_squared_error, matthews_corrcoef
from functools import partial

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
        self.c = getattr(dataset, 'c')
        self.classes = getattr(dataset, 'classes')
        self.model_type = getattr(self.model, 'model_type')
        self.model_name = getattr(self.model, 'model_name')

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

        return cv_results

    @property
    def optimization_type(self):
        if self.model_type == 'regressor':
            return 'minimize'
        return 'maximize'

    def objective(self, trial: optuna.Trial, data: PandasDataset, splitter: Splitter, run_type: str):

        """
        Objective function to run
        hyperparameter optimization on
        a dataset.

        Parameters
        ----------
        trial
            An optuna trial.
            This will be set automatically by optuna.study.optimize.
        data
            a MolDataset object.
        splitter
            splitter strategy to use.
        run_type
            Type of optimization run.
            If 'maximize', optuna will try to increase
            the performance metric.
            If 'minimize', optuna will decrease the
            performance metric.

        Returns
        -------
        score : float
            Final score of Optuna optimization.
        """

        params = {}  # type: ignore
        gather_metrics = []  # type: ignore
        estimator = self.model.model

        if hasattr(estimator, 'named_steps'):
            estimator_name = estimator.steps[-1][1].__class__.__name__
            _estimator = estimator.steps[-1][1]

        elif 'Chain' in estimator.__class__.__name__ and hasattr(estimator, 'base_estimator'):
            estimator_name = estimator.base_estimator.__class__.__name__
            _estimator = getattr(estimator, 'base_estimator')

        else:
            estimator_name = estimator.__class__.__name__

        if estimator_name in ['RandomForestRegressor', 'RandomForestClassifier']:
            params = {'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=100),
                      'max_depth': trial.suggest_int("max_depth", 2, 20, step=2),
                      'min_samples_leaf': trial.suggest_int("min_samples_leaf", 5, 20, step=5),
                      'min_samples_split': trial.suggest_int("min_samples_split", 2, 10, step=2),
                      'max_features': trial.suggest_float("max_features", 0.10, 1.0),
                      'max_samples': trial.suggest_float('max_samples', 0.25, 0.99)}

        elif estimator_name in ['XGBRegressor', 'XGBClassifier']:
            params = {"n_estimators": trial.suggest_int('n_estimators', 100, 3000, step=100),
                      'max_depth': trial.suggest_int('max_depth', 2, 20, step=2),
                      'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1e2),
                      'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1e2),
                      'min_child_weight': trial.suggest_int('min_child_weight', 1, 10, step=1),
                      'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                      'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.1),
                      }

        elif estimator_name in ['KNeighborsRegressor', 'KNeighborsClassifier']:
            params = {'n_neighbors': trial.suggest_int('n_neighbors', 5, 50, step=5)}

        elif estimator_name in ['SVR', 'SVC']:
            params = {'C': trial.suggest_loguniform('C', 1e-3, 1e3),
                      'gamma': trial.suggest_loguniform('gamma', 1e-3, 1e3),
                      'kernel': 'rbf'}

        elif estimator_name in ['LogisticRegression']:
            params = {'C': trial.suggest_loguniform('C', 1e-3, 1e3)}

        if hasattr(estimator, 'named_steps'):
            estimator.steps[-1][1].set_params(**params)

        elif 'Chain' in estimator.__class__.__name__ and hasattr(estimator, 'base_estimator'):
            estimator.base_estimator.set_params(**params)

        else:
            estimator.set_params(**params)

        score_metric = partial(mean_squared_error, squared=False) if run_type == 'minimize' else matthews_corrcoef

        # Cross-validation loop.
        logger.info(f'Hyperparameters: {params}')

        for fold, (train_split, valid_split) in tqdm(enumerate(splitter.split(data)),
                                                     total=5,
                                                     position=0,
                                                     leave=False):
            xtrain, ytrain = self.get_data(data)

            logger.info(f'Fitting on fold {fold}')
            logger.info(f'Training on {len(train_split)} samples.')
            logger.info(f'Validating on {len(valid_split)} samples.')

            self.fit(x=xtrain, y=ytrain)

            logger.info(f'Finished fold {fold}')

            preds = self.predict(self.train_data[valid_split])
            yvalid = self.train_data[valid_split][self.target_variable]

            score = score_metric(yvalid, preds)
            gather_metrics.append(score)

        avg_score = np.mean(gather_metrics)
        return avg_score

    def cross_val_optim(self,
                        splitter: Splitter,
                        n_splits: int = 10,
                        n_trials: int = 20,
                        refit: bool = True):

        """
        Helper function to run cross-validation with hyperparameter optimization using Optuna.
        If refit is True, the model will be retrained using the best hyperparameters found.

        Parameters
        ----------
        splitter
            An iterator to generate cross-validation folds.
        n_splits
            The number of cross-validation folds.
        n_trials
            The number of optimization trials
        refit
            Whether to refit the model with best hyperparameters.

        Returns
        -------
        best_params, study : dict, `optuna.study`

            Returns the best parameters and an optuna study.


        """
        run_type = self.optimization_type
        cv_iterator = CrossValidationSplitter(splitter=splitter, n_splits=n_splits)

        logger.info(
            f'Hyperparameter optimization\nEstimator: {self.model_name}\n'
            f'Number of Optuna trials: {n_trials}\n'
            f'Objective: {run_type}')
        logger.info(f'Algorithm: {self.model_name}')
        logger.info(f'Descriptors: {self.featurizer}')

        study = optuna.create_study(direction=run_type,
                                    sampler=TPESampler(),
                                    pruner=MedianPruner(n_warmup_steps=2))

        study.optimize(partial(self.objective,
                               data=self.train_data,
                               splitter=cv_iterator,
                               run_type=run_type),
                       n_trials=n_trials, n_jobs=4)

        _best_params = study.best_params

        logger.info('Study results - Dataframe')
        logger.info(study.trials_dataframe())
        logger.info('\n')

        logger.info(f'Best score:  {study.best_value}')
        logger.info(f'Best params:  {_best_params}')

        if refit:
            # Fitting best model
            x, y = self.get_data(self.train_data)
            print(f'Fitting {self.model_name} with optimized hyperparameters')
            self.fit(x=x, y=y, params=_best_params)

        return _best_params, study
