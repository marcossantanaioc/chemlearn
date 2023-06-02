__all__ = ['sklm_to_scorer', 'R2Score', 'MSEScore',
           'MAEScore', 'BalancedAccuracyScore', 'MatthewsCorrCoef',
           'PrecisionScore', 'RecallScore', 'ROCAucScore']

from functools import partial
from typing import Callable
from numpy.typing import ArrayLike

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    make_scorer, precision_score, recall_score, \
    balanced_accuracy_score, \
    matthews_corrcoef, roc_auc_score


class Metric:
    def __init__(self, metric: Callable):
        self.metric = metric

    def score(self, ytrue: ArrayLike, ypred: ArrayLike) -> float:
        return self.metric(ytrue, ypred)


def sklm_to_scorer(func, **kwargs):
    """
    Converts func into sklearn scorer
    Parameters
    ----------
    func
        a scikit-learn metric.
    Returns
    -------
        a scikit-learn scorer object.

    """

    return make_scorer(func, **kwargs)


class R2Score(Metric):
    """
    :math:`R^2` (coefficient of determination) regression score function.

    Parameters
    ----------
    transform
        Whether to transform into scorer.
    Returns
    -------
    r2
        The  score :math:`R^2` or ndarray of scores if ‘multioutput’ is ‘raw_values’.
    """

    def __init__(self, transform: bool = False):
        super().__init__(metric=r2_score)
        self.transform = transform

    @property
    def metric_func(self):
        if self.transform:
            return sklm_to_scorer(self.metric)
        return r2_score

    def score(self, ytrue: ArrayLike, ypred: ArrayLike) -> float:
        return self.metric(ytrue, ypred)


class MSEScore(Metric):
    """
    Mean squared error regression loss.

    Parameters
    ----------
    transform
        Whether to transform into scorer.
    squared
        If true, returns MSE.
        Else, returns RMSE.
    Returns
    -------
    mse
        If True returns MSE value, if False returns RMSE value.
    """

    def __init__(self, squared: bool = True, transform: bool = False):
        super().__init__(metric=mean_squared_error)
        self.squared = squared
        self.transform = transform

    @property
    def metric_func(self):
        if self.transform:
            return sklm_to_scorer(self.metric, squared=self.squared)
        return partial(self.metric, squared=self.squared)

    def score(self, ytrue: ArrayLike, ypred: ArrayLike) -> float:
        return self.metric(ytrue, ypred)


class MAEScore(Metric):
    """
    Mean absolute error regression loss.

    Parameters
    ----------
    transform
        Whether to transform into scorer.
    Returns
    -------
        mae
             returns MAE.
    """

    def __init__(self, transform: bool = False):
        super().__init__(metric=mean_absolute_error)
        self.transform = transform

    @property
    def metric_func(self):
        if self.transform:
            return sklm_to_scorer(self.metric)
        return partial(self.metric)

    def score(self, ytrue: ArrayLike, ypred: ArrayLike) -> float:
        return self.metric(ytrue, ypred)


class BalancedAccuracyScore(Metric):
    """
    Computes Matthew's correlation coefficient between actual and predicted values.
    Parameters
    ----------
    transform
        Whether to transform into scorer.
    adjusted
        When true, the result is adjusted for chance, so that random performance would
        score 0, while keeping perfect performance at a score of 1.
    Returns
    -------
    balanced_accuracy
        Balanced accuracy score.
    """

    def __init__(self, adjusted: bool = False, transform: bool = False):
        super().__init__(metric=balanced_accuracy_score)
        self.transform = transform
        self.adjusted = adjusted

    @property
    def metric_func(self):
        if self.transform:
            return sklm_to_scorer(self.metric, adjusted=self.adjusted)
        return partial(self.metric, adjusted=self.adjusted)

    def score(self, ytrue: ArrayLike, ypred: ArrayLike) -> float:
        return self.metric(ytrue, ypred)


class MatthewsCorrCoef(Metric):
    """
    Computes Matthew's correlation coefficient between actual and predicted values.
    Parameters
    ----------
    transform
        Whether to transform into scorer.
    Returns
    -------
    mcc
        The Matthews correlation coefficient (+1 represents a perfect prediction, 0 an average
        random prediction and -1 and inverse prediction).
    """

    def __init__(self, transform: bool = False):
        super().__init__(metric=matthews_corrcoef)
        self.transform = transform

    @property
    def metric_func(self):
        if self.transform:
            return sklm_to_scorer(self.metric)
        return partial(self.metric)

    def score(self, ytrue: ArrayLike, ypred: ArrayLike) -> float:
        return self.metric(ytrue, ypred)


class PrecisionScore(Metric):
    """
    Computes Precision between actual and predicted values
    Parameters
    ----------
    average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='binary'

            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
    pos_label
        The label of the positive class. Only applied to binary ``y_true``.
        For multilabel-indicator ``y_true``, ``pos_label`` is fixed to 1.

    transform
        Whether to transform into scorer.
    Returns
    -------
    precision
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task.
    """

    def __init__(self, average: str = 'binary', pos_label: int = 1, transform: bool = False):
        super().__init__(metric=precision_score)
        self.transform = transform
        self.average = average
        self.pos_label = pos_label

    @property
    def metric_func(self):
        if self.transform:
            return sklm_to_scorer(self.metric, average=self.average, pos_label=self.pos_label)
        return partial(self.metric)

    def score(self, ytrue: ArrayLike, ypred: ArrayLike) -> float:
        return self.metric(ytrue, ypred)


class RecallScore(Metric):
    """
    Computes Recall (Senstivity) between actual and predicted values
    Parameters
    ----------
    average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='binary'

        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    pos_label
        The label of the positive class. Only applied to binary ``y_true``.
        For multilabel-indicator ``y_true``, ``pos_label`` is fixed to 1.
    transform
        Whether to transform into scorer.
    Returns
    -------
    recall
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task.
    """

    def __init__(self, average: str = 'binary', pos_label: int = 1, transform: bool = False):
        super().__init__(metric=recall_score)
        self.transform = transform
        self.average = average
        self.pos_label = pos_label

    @property
    def metric_func(self):
        if self.transform:
            return sklm_to_scorer(self.metric, average=self.average, pos_label=self.pos_label)
        return partial(self.metric)

    def score(self, ytrue: ArrayLike, ypred: ArrayLike) -> float:
        return self.metric(ytrue, ypred)


class ROCAucScore(Metric):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    Parameters
    ----------
    average : {'micro', 'macro', 'samples', 'weighted'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:
        Note: multiclass ROC AUC currently only handles the 'macro' and
        'weighted' averages.
        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.
        Will be ignored when ``y_true`` is binary.
    transform
        Whether to transform into scorer.
    Returns
    -------
    auc
        Area under ROC for predicted classes
    """

    def __init__(self, average: str = 'macro', transform: bool = False):
        super().__init__(metric=roc_auc_score)
        self.transform = transform
        self.average = average

    @property
    def metric_func(self):
        if self.transform:
            return sklm_to_scorer(self.metric, average=self.average)
        return partial(self.metric)

    def score(self, ytrue: ArrayLike, ypred: ArrayLike) -> float:
        return self.metric(ytrue, ypred)
