import numpy as np
from math import ceil
from sklearn.base import BaseEstimator as be
from sklearn.base import TransformerMixin as mt


from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_array


import warnings
from cs2270Repository.utilities.piecewise import PiecewiseAggregateApproximation
from saxGeneration.symbolApprox import SymbolicAggregateApproximation


from preprocessing.scale import StandardScaler
from utilities.segmentation import windowed_view

import utilities.piecewise.PiecewiseAggregateApproximation


class BagOfWords(be, mt):

    def __init__(self, window_size=0.5, word_size=0.5, num_letters=4,
                 strat='normal', num_reduct=True, step_size=1,
                 normalized_mean=True, normalized_standard=True, overlap=True,
                 alphabet=None):
        self.window_size = window_size
        self.word_size = word_size
        self.num_letters = num_letters
        self.strat = strat
        self.num_reduct = num_reduct
        self.step_size = step_size
        self.normalized_mean = normalized_mean
        self.normalized_standard = normalized_standard
        self.overlap = overlap
        self.alphabet = alphabet

        # warnings.warn("BagOfWords has been reworked in 0.11 in order to match "
        #               "its definition in the literature. To get the old "
        #               "BagOfWords, use pyts.bag_of_words.WordExtractor "
        #               "instead.", FutureWarning)

    #fit done in another step, so just return self
    def fit(self, X, y=None):
        return self

    #transform each series into bag of words
    def transform(self, X):
        
        X = check_array(X, dtype='float64')
        samples, timestamps = X.shape
        window_size, word_size, step_size, alphabet = self.check_params(
            timestamps)
        windows = (timestamps - window_size + step_size) // step_size

        # Extract subsequences using a sliding window
        X_window = windowed_view(
            X, samples, timestamps, window_size, step_size
        ).reshape(samples * windows, window_size)

        # Create a pipeline with three steps: standardization, PAA, SAX
        pipeline = make_pipeline(
            StandardScaler(
                with_mean=self.normalized_mean, with_std=self.normalized_standard
            ),
            PiecewiseAggregateApproximation(
                window_size=None, output_size=word_size,
                overlap=self.overlap
            ),
            SymbolicAggregateApproximation(
                num_spots=self.num_spots, strat=self.strat,
                alphabet=self.alphabet
            )
        )
        X_sax = pipeline.fit_transform(X_window).reshape(
            samples, windows, word_size)

        # Join letters to make words
        X_word = np.asarray([[''.join(X_sax[i, j])
                              for j in range(windows)]
                             for i in range(samples)])

        # Apply numerosity reduction
        if self.num_reduct:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1],
                              np.full(samples, True)]
            X_bow = np.asarray([' '.join(X_word[i, not_equal[i]])
                                for i in range(samples)])
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(samples)])

        return X_bow

    def check_params(self, timestamps):
        if not isinstance(self.window_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_size' must be an integer or a float.")
        if isinstance(self.window_size, (int, np.integer)):
            if not 1 <= self.window_size <= timestamps:
                raise ValueError(
                    "If 'window_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "timestamps (got {0}).".format(self.window_size)
                )
            window_size = self.window_size
        else:
            if not 0 < self.window_size <= 1:
                raise ValueError(
                    "If 'window_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.window_size)
                )
            window_size = ceil(self.window_size * timestamps)

        if not isinstance(self.word_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'word_size' must be an integer or a float.")
        if isinstance(self.word_size, (int, np.integer)):
            if not 1 <= self.word_size <= window_size:
                raise ValueError(
                    "If 'word_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "window_size (got {0}).".format(self.word_size)
                )
            word_size = self.word_size
        else:
            if not 0 < self.word_size <= 1:
                raise ValueError(
                    "If 'word_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.word_size)
                )
            word_size = ceil(self.word_size * window_size)

        if not isinstance(self.num_spots, (int, np.integer)):
            raise TypeError("'num_spots' must be an integer.")
        if not 2 <= self.num_spots <= min(word_size, 26):
            raise ValueError(
                "'num_spots' must be greater than or equal to 2 and lower than "
                "or equal to min(word_size, 26) (got {0})."
                .format(self.num_spots)
            )

        if self.strat not in ['uniform', 'quantile', 'normal']:
            raise ValueError("'strat' must be either 'uniform', 'quantile' "
                             "or 'normal' (got {0})".format(self.strat))

        if not isinstance(self.step_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'step_size' must be an integer or a float.")
        if isinstance(self.step_size, (int, np.integer)):
            if not 1 <= self.step_size <= timestamps:
                raise ValueError(
                    "If 'step_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "timestamps (got {0}).".format(self.step_size)
                )
            step_size = self.step_size
        else:
            if not 0 < self.step_size <= 1:
                raise ValueError(
                    "If 'step_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.step_size)
                )
            step_size = ceil(self.step_size * timestamps)

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
                raise ValueError("If 'alphabet' is array-like, its shape ")
