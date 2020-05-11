"""Code for Symbolic Aggregate approXimation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array


class SymbolicAggregateApproximation(BaseEstimator, TransformerMixin):
    """Symbolic Aggregate approXimation.

    Parameters
    ----------
    num_letters : int (default = 4)
        The number of bins to produce. It must be between 2 and
        ``min(num_stamps, 26)``.

    strategy : 'uniform', 'quantile' or 'normal' (default = 'quantile')
        strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution

    alphabet : None, 'ordinal' or array-like, shape = (num_letters,)
        Alphabet to use. If None, the first `num_letters` letters of the Latin
        alphabet are used. If 'ordinal', integers are used.

    References
    ----------
    .. [1] J. Lin, E. Keogh, L. Wei, and S. Lonardi, "Experiencing SAX: a
           novel symbolic representation of time series". Data Mining and
           Knowledge Discovery, 15(2), 107-144 (2007).

    Examples
    --------
    # >>> from pyts.approximation import SymbolicAggregateApproximation
    # >>> X = [[0, 4, 2, 1, 7, 6, 3, 5],
    # ...      [2, 5, 4, 5, 3, 4, 2, 3]]
    # >>> transformer = SymbolicAggregateApproximation()
    # >>> print(transformer.transform(X))
    # [['a' 'c' 'b' 'a' 'd' 'd' 'b' 'c']
    #  ['a' 'd' 'c' 'd' 'b' 'c' 'a' 'b']]
    #
    # """

    def __init__(self, num_letters=4, strategy='quantile', alphabet=None):
        self.num_letters = num_letters
        self.strategy = strategy
        self.alphabet = alphabet

    def fit(self, X=None, y=None):
       #fit ignored in SAX... only used for transform
        return self

    def transform(self, X):
        
        X = check_array(X, dtype="float64")
        num_stamps = X.shape[1]

        alphabet = self._check_params(self.num_letters, num_stamps)
        discretizer = kbd(num_letters=self.num_letters, strategy=self.strategy)

        index = discretizer.fit_transform(X)

        if not isinstance(alphabet, str):
            return alphabet[index]
        else:
            return index

    def check_params(self, num_stamps):
        if not isinstance(self.num_letters, (int, np.integer)):
            raise TypeError("'num_letters' must be an integer.")
        if not 2 <= self.num_letters <= min(num_stamps, 26):
            raise ValueError(
                "'num_letters' must be greater than or equal to 2 and lower than "
                "or equal to min(num_stamps, 26) (got {0})."
                .format(self.num_letters)
            )
        if self.strategy not in ['uniform', 'quantile', 'normal']:
            raise ValueError("'strategy' must be either 'uniform', 'quantile' "
                             "or 'normal' (got {0})".format(self.strategy))
        if not ((self.alphabet is None)
                or (self.alphabet == 'ordinal')
                or (isinstance(self.alphabet, (list, tuple, np.ndarray)))):
            raise TypeError("'alphabet' must be None, 'ordinal' or array-like "
                            "with shape (num_letters,) (got {0})"
                            .format(self.alphabet))
        if self.alphabet is None:
            alphabet = np.array([chr(i) for i in range(97, 97 + self.num_letters)])
        elif self.alphabet == 'ordinal':
            alphabet = 'ordinal'
        else:
            alphabet = check_array(self.alphabet, ensure_2d=False, dtype=None)
            if alphabet.shape != (self.num_letters,):
                raise ValueError("If 'alphabet' is array-like, its shape "
                                 "must be equal to (num_letters,).")
        return alphabet
