from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbsScaler
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.utils.validation import check_array


class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize time series by removing mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : bool (default = True)
        If True, center the data before scaling.

    with_std : bool (default = True)
        If True, scale the data to unit variance.

    Examples
    --------
    # >>> from pyts.preprocessing import StandardScaler
    # >>> X = [[0, 2, 0, 4, 4, 6, 4, 4],
    # ...      [1, 0, 3, 2, 2, 2, 0, 2]]
    # >>> scaler = StandardScaler()
    # >>> scaler.transform(X)
    # array([[-1.5, -0.5, -1.5,  0.5,  0.5,  1.5,  0.5,  0.5],
    #        [-0.5, -1.5,  1.5,  0.5,  0.5,  0.5, -1.5,  0.5]])
    #
    # """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X=None, y=None):
        "fit not necessary here"
        return self

    def transform(self, X):
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to scale.

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_timestamps)
            Scaled data.

        """
        X = check_array(X, dtype='float64')
        scaler = SklearnStandardScaler(
            with_mean=self.with_mean, with_std=self.with_std)
        X_new = scaler.fit_transform(X.T).T
        return X_new


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Transforms samples by scaling each sample to a given range.

    Parameters
    ----------
    sample_range : tuple (min, max) (default = (0, 1))
        Desired range of transformed data.

    Examples
    --------
    # >>> from pyts.preprocessing import MinMaxScaler
    # >>> X = [[1, 5, 3, 2, 9, 6, 4, 7],
    # ...      [1, -2, 3, 2, 2, 1, 0, 2]]
    # >>> scaler = MinMaxScaler()
    # >>> scaler.transform(X)
    # array([[0.   , 0.5  , 0.25 , 0.125, 1.   , 0.625, 0.375, 0.75 ],
    #        [0.6  , 0.   , 1.   , 0.8  , 0.8  , 0.6  , 0.4  , 0.8  ]])
    #
    # """

    def __init__(self, sample_range=(0, 1)):
        self.sample_range = sample_range

    def fit(self, X=None, y=None):
        "fit unncessary"
        return self

    def transform(self, X):
        """Scale samples of X according to sample_range.

        input: X = (samples, timestamps)
        output: X = (samples, timestamps);
        """

        X = check_array(X, dtype='float64')
        scaler = SklearnMinMaxScaler(feature_range=self.sample_range)
        X_new = scaler.fit_transform(X.T).T
        return X_new


class MaxAbsScaler(BaseEstimator, TransformerMixin):
    """Scale each sample by its maximum absolute value.

    Examples
    --------
    # >>> from pyts.preprocessing import MaxAbsScaler
    # >>> X = [[1, 5, 3, 2, 10, 6, 4, 7],
    # ...      [1, -5, 3, 2, 2, 1, 0, 2]]
    # >>> scaler = MaxAbsScaler()
    # >>> scaler.transform(X)
    # array([[ 0.1,  0.5,  0.3,  0.2,  1. ,  0.6,  0.4,  0.7],
    #        [ 0.2, -1. ,  0.6,  0.4,  0.4,  0.2,  0. ,  0.4]])
    #
    # """

    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X):
        """Scale the data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to scale.

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_timestamps)
            Scaled data.

        """
        X = check_array(X, dtype='float64')
        scaler = SklearnMaxAbsScaler()
        X_new = scaler.fit_transform(X.T).T
        return X_new
