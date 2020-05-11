
import numpy as np
import os
import pickle
from scipy.io.arff import loadarff
from sklearn.utils import Bunch
from urllib.request import urlretrieve
import zipfile

def _load_ucr_dataset(dataset, path):
    """Load a UCR data set from a local folder.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    path : str
        The path of the folder containing the cached data set.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array
            The classification labels in the training set.
        target_test : array
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    Notes
    -----
    Padded values are represented as NaN's.

    """
    new_path = path + dataset + '/'
    try:
        with(open(new_path + dataset + '.txt', encoding='utf-8')) as f:
            description = f.read()
    except UnicodeDecodeError:
        with(open(new_path + dataset + '.txt', encoding='ISO-8859-1')) as f:
            description = f.read()
    try:
        data_train = np.genfromtxt(new_path + dataset + '_TRAIN.txt')
        data_test = np.genfromtxt(new_path + dataset + '_TEST.txt')

        X_train, y_train = data_train[:, 1:], data_train[:, 0]
        X_test, y_test = data_test[:, 1:], data_test[:, 0]

    except IndexError:
        train = loadarff(new_path + dataset + '_TRAIN.arff')
        test = loadarff(new_path + dataset + '_TEST.arff')

        data_train = np.asarray([train[0][name] for name in train[1].names()])
        X_train = data_train[:-1].T.astype('float64')
        y_train = data_train[-1]

        data_test = np.asarray([test[0][name] for name in test[1].names()])
        X_test = data_test[:-1].T.astype('float64')
        y_test = data_test[-1]

    try:
        y_train = y_train.astype('float64').astype('int64')
        y_test = y_test.astype('float64').astype('int64')
    except ValueError:
        pass

    bunch = Bunch(
        data_train=X_train, target_train=y_train,
        data_test=X_test, target_test=y_test,
        DESCR=description,
        url=("http://www.timeseriesclassification.com/"
             "description.php?Dataset={}".format(dataset))
    )

    return bunch

def _load_dataset(name, archive, return_X_y):
    r"""Load and return dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.

    archive : 'UCR' or 'UEA'
        Archive the dataset belongs to.

    return_X_y : bool
        If True, return
        ``(data_train, data_test, target_train, target_test)`` instead of a
        Bunch object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array of integers
            The classification labels in the training set.
        target_test : array of integers
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    (data_train, data_test, target_train, target_test) : tuple if ``return_X_y`` is True

    """  # noqa: E501
    module_path = os.path.dirname(__file__)
    folder = os.path.join(module_path, 'cached_datasets', archive, '')
    if archive == 'UCR':
        bunch = _load_ucr_dataset(name, folder)
    # else:
    #     bunch = _load_uea_dataset(name, folder)
    if return_X_y:
        return (bunch.data_train, bunch.data_test,
                bunch.target_train, bunch.target_test)
    return bunch

def load_gunpoint(return_X_y=False):
    r"""Load and return the GunPoint dataset.

    This dataset involves one female actor and one male actor making a motion
    with their hand. The two classes are: Gun-Draw and Point: For Gun-Draw the
    actors have their hands by their sides. They draw a replicate gun from a
    hip-mounted holster, point it at a target for approximately one second,
    then return the gun to the holster, and their hands to their sides. For
    Point the actors have their gun by their sides. They point with their index
    fingers to a target for approximately one second, and then return their
    hands to their sides. For both classes, we tracked the centroid of the
    actor's right hands in both X- and Y-axes, which appear to be highly
    correlated. The data in the archive is just the X-axis.

    ================   ==============
    Training samples               50
    Test samples                  150
    Timestamps                    150
    Classes                         2
    ================   ==============

    Parameters
    ----------
    return_X_y : bool (default = False)
        If True, return
        ``(data_train, data_test, target_train, target_test)`` instead of a
        Bunch object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array of integers
            The classification labels in the training set.
        target_test : array of integers
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    (data_train, data_test, target_train, target_test) : tuple if ``return_X_y`` is True

    References
    ----------
    .. [1] `UCR archive entry for the PigCVP dataset
           <http://www.timeseriesclassification.com/description.php?Dataset=GunPoint>`_

    Examples
    --------
    # # >>> from pyts.datasets import load_gunpoint
    # # >>> bunch = load_gunpoint()
    # # >>> bunch.data_train.shape
    # # (50, 150)
    # # >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    # # >>> X_train.shape
    # # (50, 150)
    # 
    # """  # noqa: E501
    return _load_dataset('GunPoint', 'UCR', return_X_y)

