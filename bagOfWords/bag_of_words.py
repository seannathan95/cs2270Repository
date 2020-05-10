import numpy as np
from math import ceil
from sklearn.base import BaseEstimator as estimator
from sklearn.base import TransformMixin as mixTransform
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_array
import warnings
from ..approximation import (
    PiecewiseAggregateApproximation, SymbolicAggregateApproximation)
from ..preprocessing import StandardScaler
from ..utils.utils import _windowed_view


class BagOfWords(estimator, mixTransform) {
    # def __init__self(window_len = 20, step_size = 1, num_spots = 4,
    #                 strat = 'normal', num_reduct = True,
    #                 word_len = 5, normalized_standard = True, overlap = True,
    #                 alphabet = None):
    #     self.window_len = window_len
    #     self.step_size = step_size
    #     self.num_reduct = num_reduct
    #     self.word_len = word_len
    #     self.strat = strat
    #     self.num_spots = num_spots

    # #no fit utilized here.. comes from previous steps
    # def fit(self X=None, y=None):

    #     return self
    
    # def transform(self, X):
    #     X = check_array(X, dtype=None)
    #     samples, stamps = self.check_params(stamps)

    def __init__(self, window_len=0.5, word_len=0.5, num_spots=4,
                 strat='normal', num_reduct=True, step_size=1,
                 normalized_mean=True, normalized_standard=True, overlap=True,
                 alphabet=None):
        self.window_len = window_len
        self.word_len = word_len
        self.num_spots = num_spots
        self.strat = strat
        self.num_reduct = num_reduct
        self.step_size = step_size
        self.normalized_mean = normalized_mean
        self.normalized_standard = normalized_standard
        self.overlap = overlap
        self.alphabet = alphabet

        warnings.warn("BagOfWords has been reworked in 0.11 in order to match "
                      "its definition in the literature. To get the old "
                      "BagOfWords, use pyts.bag_of_words.WordExtractor "
                      "instead.", FutureWarning)

    #fit done in another step, so just return self
    def fit(self, X, y=None):
        return self

    #transform each series into bag of words
    def transform(self, X):
        
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape
        window_len, word_len, step_size, alphabet = self._check_params(
            n_timestamps)
        n_windows = (n_timestamps - window_len + step_size) // step_size

        # Extract subsequences using a sliding window
        X_window = _windowed_view(
            X, n_samples, n_timestamps, window_len, step_size
        ).reshape(n_samples * n_windows, window_len)

        # Create a pipeline with three steps: standardization, PAA, SAX
        pipeline = make_pipeline(
            StandardScaler(
                with_mean=self.normalized_mean, with_std=self.normalized_standard
            ),
            PiecewiseAggregateApproximation(
                window_len=None, output_size=word_len,
                overlap=self.overlap
            ),
            SymbolicAggregateApproximation(
                num_spots=self.num_spots, strat=self.strat,
                alphabet=self.alphabet
            )
        )
        X_sax = pipeline.fit_transform(X_window).reshape(
            n_samples, n_windows, word_len)

        # Join letters to make words
        X_word = np.asarray([[''.join(X_sax[i, j])
                              for j in range(n_windows)]
                             for i in range(n_samples)])

        # Apply numerosity reduction
        if self.num_reduct:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1],
                              np.full(n_samples, True)]
            X_bow = np.asarray([' '.join(X_word[i, not_equal[i]])
                                for i in range(n_samples)])
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

        return X_bow

    def _check_params(self, n_timestamps):
        if not isinstance(self.window_len,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_len' must be an integer or a float.")
        if isinstance(self.window_len, (int, np.integer)):
            if not 1 <= self.window_len <= n_timestamps:
                raise ValueError(
                    "If 'window_len' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.window_len)
                )
            window_len = self.window_len
        else:
            if not 0 < self.window_len <= 1:
                raise ValueError(
                    "If 'window_len' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.window_len)
                )
            window_len = ceil(self.window_len * n_timestamps)

        if not isinstance(self.word_len,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'word_len' must be an integer or a float.")
        if isinstance(self.word_len, (int, np.integer)):
            if not 1 <= self.word_len <= window_len:
                raise ValueError(
                    "If 'word_len' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "window_len (got {0}).".format(self.word_len)
                )
            word_len = self.word_len
        else:
            if not 0 < self.word_len <= 1:
                raise ValueError(
                    "If 'word_len' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.word_len)
                )
            word_len = ceil(self.word_len * window_len)

        if not isinstance(self.num_spots, (int, np.integer)):
            raise TypeError("'num_spots' must be an integer.")
        if not 2 <= self.num_spots <= min(word_len, 26):
            raise ValueError(
                "'num_spots' must be greater than or equal to 2 and lower than "
                "or equal to min(word_len, 26) (got {0})."
                .format(self.num_spots)
            )

        if self.strat not in ['uniform', 'quantile', 'normal']:
            raise ValueError("'strat' must be either 'uniform', 'quantile' "
                             "or 'normal' (got {0})".format(self.strat))

        if not isinstance(self.step_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'step_size' must be an integer or a float.")
        if isinstance(self.step_size, (int, np.integer)):
            if not 1 <= self.step_size <= n_timestamps:
                raise ValueError(
                    "If 'step_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.step_size)
                )
            step_size = self.step_size
        else:
            if not 0 < self.step_size <= 1:
                raise ValueError(
                    "If 'step_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.step_size)
                )
            step_size = ceil(self.step_size * n_timestamps)

        if not ((self.alphabet is None)
                or (isinstance(self.alphabet, (list, tuple, np.ndarray)))):
            raise TypeError("'alphabet' must be None or array-like "
                            "with shape (num_spots,) (got {0})."
                            .format(self.alphabet))
        if self.alphabet is None:
            alphabet = np.array([chr(i) for i in range(97, 97 + self.num_spots)])
        else:
            alphabet = check_array(self.alphabet, ensure_2d=False, dtype=None)
            if alphabet.shape != (self.num_spots,):
                raise ValueError("If 'alphabet' is array-like, its shape "
}