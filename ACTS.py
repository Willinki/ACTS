"""
Implementation of the ACTS algorithm as a query strategy for the va_builder framework.
"""
import numpy as np
from modAL.utils.data import modALinput 
from sklearn.base import BaseEstimator


def _shuffled_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Taken for modAL.utils
    Shuffles the values and sorts them afterwards. This can be used to break
    the tie when the highest utility score is not unique. The shuffle randomizes
    order, which is preserved by the mergesort algorithm.
    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices to return.
    Returns:
        The indices of the n_instances largest values.
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

    # shuffling indices and corresponding values
    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]

    # getting the n_instances best instance
    # since mergesort is used, the shuffled order is preserved
    sorted_query_idx = np.argsort(shuffled_values, kind='mergesort')[len(shuffled_values)-n_instances:]

    # inverting the shuffle
    query_idx = shuffled_idx[sorted_query_idx]
    return query_idx


def _multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Taken from modAL.utils
    Selects the indices of the n_instances highest values.
    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices to return.
    Returns:
        The indices of the n_instances largest values.
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

    max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
    return max_idx


def _initialize_patterns(DL):
    print("Yup... This initializes the patterns")


def _update_patterns(Opatterns, DL):
    print("What? Patterns already initialized? Let's update then")
    
# SHOULD RETURN UNCERTAINTY VALUE
def compute_uncertainty(DL, X, L, k):
    # uncertainty =

    # return uncertainty

def compute_utility(DU, n, S):
    # COMPUTE UTILITY HERE



def ACTS(classifier: BaseEstimator, X: modALinput,
         # HERE I'LL ADD THE ADDITIONAL REQUIREMENTS
         DL: modALinput, L: np.ndarray, 
         # THESE ELEMENTS ARE HERE BY DEFAULT
         n_instances: int = 1, random_tie_break: bool = False,
         **uncertainty_measure_kwargs) -> np.ndarray :
    """
    Sampling based on the measures defined by ACTS.
    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        DL: The labelled data
        L: The labels of DL
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.
    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    #
    # actual algorithm
    #
    #
    #           CALCULATES the question informativeness for each element in X
    #
    #
    # PROBLEM N° 1: the algorithm needs also the labelled examples
    #               They must be passed as the uncertainty_measure_kwargs
    #               during the query part
    # PROBLEM N° 2: it also needs the set of possible labels, same for problem 1
    # ----> SOLUTION: in utils.alutils.select_query_strategy it is possible to add 
    #                 another condition where if ACTS is used, all the requirements 
    #                 are passed with "partial()"
    # PROBLEM n° 3: patterns might be a static variable (stored even when they are 
    #               out of scope)
    # ----> SOLUTION (thanks StackOverflow): 
    try:
        ACTS.patterns = _update_patterns(ACTS.patterns, DL)
    except AttributeError:
        ACTS.patterns = _initialize_patterns(DL)
    # 
    # STEPS:
    #        initialize/update patterns (above)
    #        compute utility
    #        compute uncertainty
    #        if uncertainty = 0, add the series to the labelled set (is it possible?)[maybe leave it as last...]
    #        calculate Q_informativeness 
    if not random_tie_break:
        return _multi_argmax(Q_informativeness, n_instances=n_instances)

    return _shuffled_argmax(Q_informativeness, n_instances=n_instances)
