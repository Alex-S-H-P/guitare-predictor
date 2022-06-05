

import sklearn.ensemble._forest as f
from sklearn.ensemble import RandomForestClassifier
import threading
import numpy as np
from joblib import Parallel
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils.fixes import delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted


def simple_note(n: str):
    return n.partition(':')[0]

def notes_by_key(key: str):
    C_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    Csharp_notes = ['C#', 'Eb', 'F', 'F#', 'Ab', 'Bb', 'B#']
    D_notes = ['D', 'E', 'F#', 'G', 'A', 'B', 'C#']
    Eb_notes = ['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D']
    E_notes = ['E', 'F#', 'Ab', 'A', 'B', 'C#', 'Eb']
    F_notes = ['F', 'G', 'A', 'Bb', 'C', 'D', 'E']
    Fsharp_notes = ['F#', 'Ab', 'Bb', 'B', 'C#', 'Eb', 'F']
    G_notes = ['G', 'A', 'B', 'C', 'D', 'E', 'F#']
    Ab_notes = ['Ab', 'Bb', 'C', 'C#', 'Eb', 'F', 'G']
    A_notes = ['A', 'B', 'C#', 'D', 'E', 'F#', 'Ab']
    Bb_notes = ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A']
    B_notes = ['B', 'C#', 'Eb', 'E', 'F#', 'Ab', 'Bb']

    if key=='C':
        return C_notes
    elif key=='C#':
        return Csharp_notes
    elif key=='D':
        return D_notes
    elif key=='Eb':
        return Eb_notes
    elif key=='E':
        return E_notes
    elif key=='F':
        return F_notes
    elif key=='F#':
        return Fsharp_notes
    elif key=='G':
        return G_notes
    elif key=='Ab':
        return Ab_notes
    elif key=='A':
        return A_notes
    elif key=='Bb':
        return Bb_notes
    else:
        return B_notes


def key_estimation(Y):
    keys = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    key_it = [0]*12

    for note in Y:
        for i in range(12):
            if simple_note(note) in notes_by_key(keys[i]):
                key_it[i]+=1
    return keys[key_it.index(max(key_it))]


def corr_keys(all_predictions: list,key_correlations: dict):
    corr = np.zeros(len(all_predictions))
    for i in range(corr.shape[0]):
        corr[i] = key_correlations.get(key_estimation(all_predictions[i]))
    return corr


def predict_proba_with_key(self: RandomForestClassifier, all_predictions: list, X, key_correlations):
    """
    Predict class probabilities for X considering the estimated key of X.

    The predicted class probabilities of an input sample are computed as
    the mean predicted class probabilities of the trees in the forest.
    The class probability of a single tree is the fraction of samples of
    the same class in a leaf.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples. Internally, its dtype will be converted to
        ``dtype=np.float32``. If a sparse matrix is provided, it will be
        converted into a sparse ``csr_matrix``.

    Returns
    -------
    p : ndarray of shape (n_samples, n_classes), or a list of such arrays
        The class probabilities of the input samples. The order of the
        classes corresponds to that in the attribute :term:`classes_`.
    """
    check_is_fitted(self)
    # Check data
    X = self._validate_X_predict(X)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

    # avoid storing the output of every estimator by summing them here
    all_proba = [
        np.zeros((X.shape[0], j), dtype=np.float64)
        for j in np.atleast_1d(self.n_classes_)
    ]
    lock = threading.Lock()
    Parallel(
        n_jobs=n_jobs,
        verbose=self.verbose,
        **_joblib_parallel_args(require="sharedmem"),
    )(
        delayed(f._accumulate_prediction)(e.predict_proba, X, all_proba, lock)
        for e in self.estimators_
    )

    # get the correlation coefficient of the keys associated with each prediction
    coeffs = corr_keys(all_predictions, key_correlations)

    i=0
    for proba in all_proba:
        proba *= coeffs[i]
        proba /= np.sum(coeffs)
        i+=1

    if len(all_proba) == 1:
        return all_proba[0]
    else:
        return all_proba


def predict_with_key(self: RandomForestClassifier, all_predictions: list, X, key_correlations):
    """
    Predict class for X.

    The predicted class of an input sample is a vote by the trees in
    the forest, weighted by their probability estimates. That is,
    the predicted class is the one with highest mean probability
    estimate across the trees.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples. Internally, its dtype will be converted to
        ``dtype=np.float32``. If a sparse matrix is provided, it will be
        converted into a sparse ``csr_matrix``.

    Returns
    -------
    y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        The predicted classes.
    """
    proba = predict_proba_with_key(self, all_predictions, X, key_correlations)

    if self.n_outputs_ == 1:
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    else:
        n_samples = proba[0].shape[0]
        # all dtypes should be the same, so just take the first
        class_type = self.classes_[0].dtype
        predictions = np.empty((n_samples, self.n_outputs_), dtype=class_type)

        for k in range(self.n_outputs_):
            predictions[:, k] = self.classes_[k].take(
                np.argmax(proba[k], axis=1), axis=0
            )

        return predictions

