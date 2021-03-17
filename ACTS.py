"""
Implementation of the ACTS algorithm as a query strategy for the va_builder framework.
For any additional information see documentation.
"""
import pandas as pd
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


def k(X : np.array) -> int:
    """
    Extracts key from sequence of values.
    Used to assign a key to patterns.
    """
    return hash(
        round(np.mean(X), 6)
        )

    
# SHOULD RETURN UNCERTAINTY VALUE
def compute_uncertainty(DL, X, L, k):
    # uncertainty =
    return 
    # return uncertainty


def compute_utility(DU, n, S):
    # COMPUTE UTILITY HERE
    return


class ACTS:
    """Wrapper class for ACTS query strategy. 
    
    Properties
    ----------
        - patterns: pd.DataFrame
            cols : key, ts, inst_keys, labels, l_probas
            
        - instances : pd.Dataframe
            cols : key, ts, label, near_pt
    """
    def __init__(self):
        self.patterns = None
        self.instances = None
        

    def __call__(self, X: modALinput,
                 DL: modALinput, 
                 L: np.ndarray, 
                 Li: np.ndarray, 
                 n_instances: int = 1, 
                 random_tie_break: bool = False,
                 **uncertainty_measure_kwargs) -> np.ndarray :
        """Sampling based on the measures defined by ACTS.
        
        Args
        ----
            - X : The pool of samples to query from.
            - DL: The instances of labelled data
            - L: The labels of DL
            - Li: Indices of labelled instances
            - n_instances: Number of samples to be returned.
            - random_tie_break: If True, shuffles utility scores to randomize the order. This
                can be used to break the tie when the highest utility score is not unique.
            - **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
                measure function.
                
        Returns
        -------
            The indices of the instances from X chosen to be labelled;
        """
        # MAINTAINING PATTERNS
        if self.patterns is None:
            self._initialize_instances(DL, L, Li)
            self._initialize_patterns()
            self._assign_instances(empty_only=False)
        else:
            self._update_instances(DL, L, Li)
            self._assign_instances(empty_only=True)
            self._assign_patterns()
            self._update_patterns()
            self._assign_instances(empty_only=False)
            self._assign_patterns()
        # MODELING
        self._update_l_probas()
        lam = self._calculate_lambda(X)
        # QUESTION SELECTION
        # STEPS:
        #        compute utility
        #        compute uncertainty
        #        if uncertainty = 0, add the series to the labelled set (is it possible?)[maybe leave it as last...]
        #        calculate Q_informativeness 

        if not random_tie_break:
            return _multi_argmax(Q_informativeness, n_instances=n_instances)

        return _shuffled_argmax(Q_informativeness, n_instances=n_instances)

    
    def _initialize_instances(self, DL, L, Li) -> None:
        """For each element in DL, L, Li add instance
        
        Args : see __call__
        """


    def _initialize_patterns(self) -> None:
        """For each instance, add pattern
        """


    def _update_instances(self, DL, L, Li) -> None:
        """For each element in DL, check if exists in instances
           if not, add
        
        Args : see __call__
        """


    def _assign_instances(self, empty_only : bool) -> None:
        """For each instance, update near_pt
        
        Args : 
            empty_only : (bool) if true, only instances with n_pt = None are
                updated 
        """


    def _assign_patterns(self) -> None:
        """For each pattern, update inst_keys, labels
        """
        

    def _update_patterns(self) -> None:
        """For each pattern, check if mixed, 
           if yes, split (delete old pattern, add 2 new ones)
        """

        
    def _calculate_lambda(self, X):
        """Calculates the value of lambda (MLE), used in P(X | pt)
        
        Args : see __call__
        """


    @staticmethod
    def _pt_proba(X : np.array, pt : np.array, lam : float) -> float:
        """Given instance and pattern, calculates P(X | pt)
        """

    
    def _update_l_probas(self) -> None:
        """For each pt, calculate P(pt | ell) for each ell
        """