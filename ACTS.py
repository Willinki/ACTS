"""
Implementation of the ACTS algorithm as a query strategy for the va_builder framework.
For any additional information see documentation.
"""
import pandas as pd
import numpy as np
import random
import math
from numba import njit, prange
from sklearn.neighbors import NearestNeighbors
from tslearn.shapelets import LearningShapelets, grabocka_params_to_shapelet_size_dict
from tensorflow.keras.optimizers import Adam

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


def k(X: np.ndarray) -> int:
    """
    Extracts key from sequence of values.
    Used to assign a key to patterns.
    """
    return hash(
        round(np.mean(X), 6)
        )


@njit(parallel=True)
def _dis(X: np.ndarray, pt: np.ndarray) -> float:
    """Given instance and pattern, calculates Dis(X, pt), sliding window.
    Used in _calculate_probx. 
    
    Args
    ----
        - X : (array-like) instance
        - pt : (array-like) pattern
    """
    m = len(pt)
    n = len(X)
    print(n, m)
    assert n >= m, "In _dis, a sequence is longer than a pattern"
    dist_array = np.empty(shape=(n-m+1, ))
    for i in prange(0, n-m+1):
        dist_array[i] = np.linalg.norm(
            X[i:(i+m)] - pt[:]
        )
    return dist_array.min()
        

@njit(parallel=True)
def _fast_lambda(tss : np.ndarray, pts : np.ndarray) -> float:
    """Wrapper function used in ACTS._calculate_lambda
    Calculates mean of _dis(ts, pt)^(-1) for every possible ts in tss
    and pt in pts
    
    Args
    ----
        - tss : (n_instances, n_timestamps)
            2d array of instances
        - pts : (n_instances, n_timestamps)
            2d array of patterns
    
    Returns
    -------
        - lam : float
            Mean of _dis(ts, pt) between al ts in tss and pt in pts
    """
    mean = 0
    N = tss.shape[0]*pts.shape[0]
    for i in prange(tss.shape[0]):
        for j in prange(pts.shape[0]):
            mean += _dis(tss[i], pts[j])/N
    lam = 1/mean
    return lam


@njit(parallel=True)
def _fast_nn(tss: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # TODO test
    # NOTE: If instances and pattens are too many this becomes prohibitive on memory, but its the fastest
    """Wrapper function used in ACTS.assign_instances.
    For each ts in tss computes the nearest pt in pts.
    
    Args
    ----
        - tss : (n_instances, n_timestamps)
            2d array of instances
        - pts : (n_instances, n_timestamps)
            2d array of patterns
    
    Returns
    -------
        - nn_pt : array (int)
            Array of integers containing integer index of nn
            for each instance
    """
    distances = np.empty(shape=(tss.shape[0], pts.shape[0]))
    nn_pt = np.empty(shape=(tss.shape[0], ))
    for i in prange(tss.shape[0]):
        for j in prange(pts.shape[0]):
            distances[i, j] = _dis(tss[i], pts[j])
    for i in prange(tss.shape[0]):
        nn_pt[i] = np.argmin(distances[i, :])
    return nn_pt


class ACTS:
    """Wrapper class for ACTS query strategy. 
    
    Properties
    ----------
        - patterns: pd.DataFrame
            cols : key, ts, inst_keys, labels, l_probas
            
        - instances : pd.Dataframe
            cols : key, ts, label, near_pt
            
        - lam : float
            parameter for exponential distribution 
    """
    def __init__(self):
        self.patterns = None
        self.instances = None
        self.lam = None

    def __call__(self, X: np.ndarray,
                 DL: np.ndarray, 
                 L: np.ndarray, 
                 Li: np.ndarray, 
                 n_instances: int = 1, 
                 random_tie_break: bool = False,
                 **uncertainty_measure_kwargs) -> np.ndarray :
        # TODO test (do I have to say it?) 
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
        self.lam = self._calculate_lambda(X, DL)
        self._calculate_multinomial(DL)
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
        # TODO test
        """For each element in DL, L, Li add instance
        
        Args : see __call__
        """
        self.instances = pd.DataFrame({
            "key" : Li,
            "ts" : [DL[i] for i in range(DL.shape[0])],
            "label" : L,
            "near_pt" : [np.nan for _ in L]
        }).set_index("key")

    def _initialize_patterns(self) -> None:
        # TODO test
        """For each instance, add pattern
        """
        # key, ts, inst_keys, labels, l_probas
        inst_values = self.instances["ts"].to_numpy()
        self.patterns = pd.DataFrame({
           "key" : [k(inst) for inst in inst_values],
           "ts" : [inst for inst in inst_values],
           "inst_keys" : [np.array([ind]) for ind in self.instances.index],
           "labels" : [np.array([lab]) for lab in self.instances["label"].to_numpy()],
           "l_probas" : [np.nan for _ in self.instances.index]
        }).set_index("key")

    def _assign_instances(self, empty_only : bool) -> None:
        # TODO test extensively 
        """For each instance, update near_pt
        
        Args : 
            empty_only : (bool) if true, only instances with n_pt = np.nan are
                updated 
        """
        if empty_only:
            indexes = self.instances[self.instances["near_pt"].isna()].index
        else:
            indexes = self.instances.index
        patterns_array = self.patterns["ts"].to_numpy()
        int_pattern_idx = _fast_nn(
            tss=self.instances.loc[indexes]["ts"].to_numpy(),
            pts=patterns_array
        )
        # hash int indexes to pattern keys
        self.instances.loc[indexes] = [
            hash(pt) 
            for pt in patterns_array[int_pattern_idx]
        ]

    def _assign_patterns(self) -> None:
        """For each pattern, update inst_keys, labels.
        
        - self.patterns.inst_keys : np.array 
            Keys of instances that have the pattern as near_pt
        - self.patterns.labels : np.array 
            Keys of instances that have the pattern as near_pt
        """
        # TODO test
        for index, _ in self.patterns.iterrows():
            nn_instances = self.instances[
                    self.instances["near_pt"] is index
            ]
            self.pattern.loc[index]["inst_keys"] = np.array(
                nn_instances.index
            )
            self.pattern.loc[index]["labels"] = nn_instances["label"].to_numpy()

    def _update_patterns(self) -> None:
        """For each pattern, check if mixed, 
           if yes, split (delete old pattern, add new ones)
        """
        self.patterns["mixed_bool"] = self.patterns["labels"].apply(
            lambda x : len(np.unique(x))
        )
        mixed_pts = self.patterns[["inst_keys", "labels"]]
        #dropping mixed patterns
        self.patterns = self.patterns.drop(mixed_pts.index)
        # this will contain the new patterns
        new_pts = []
        for _, row in mixed_pts.iterrows():
            instances = self.instances.loc[row["inst_keys"]].to_numpy()
            labels = row["labels"]
            dict_shapelets = grabocka_params_to_shapelet_size_dict(
                n_ts = instances.shape[0], 
                ts_sz = len(instances[0]),
                n_classes = len(np.unique(labels)),
                l=0.1,
                r=1
            )
            model = LearningShapelets(
                n_shapelets_per_size=dict_shapelets,
                optimizer=Adam(.01),
                batch_size=32,
                weight_regularizer=.01,
                random_state=0,
                verbose=0
            )
            model.fit(instances, labels)
            new_pts.append(model.shapelets_as_time_series_)
        
        #ADD NEW PATTERNS
        self.patterns = self,.patterns.append(
            pd.DataFrame({
                "key" : [k(x) for x in new_pts],
                "ts" : new_pts,
                "inst_keys" : [np.nan for _ in new_pts], 
                "labels" : [np.nan for _ in new_pts], 
                "l_probas" : [np.nan for _ in new_pts]
            })
        )
            

    def _update_instances(self, DL, L, Li) -> None:
        # TODO test
        """For each element in DL, check if exists in instances
           if not, add to self.istances
        
        Args : see __call__
        """
        new_keys = np.setdiff1d(Li, self.instances.index)
        new_idxs = np.where(
            np.in1d(Li, new_keys, assume_unique=True)
        )
        self.instances = self.instances.append(
            pd.DataFrame({
                "key" : new_keys,
                "ts" : DL[new_idxs],
                "label" : L[new_idxs],
                "near_pt" : [np.nan for _ in L]
            })
        )

    def _calculate_lambda(self, X : np.ndarray, DL : np.ndarray, 
                          sample_size : float = 0.01, N : int = 50) -> None:
        # TODO test extensively
        """Calculates the value of self.lam, used in P(X | pt)
        
        Args
        ----
            - X : (shape = (n_instances, n_timestamps)
                Dataset of unlabelled instances
            - DL : (shape = (n_instances, n_timestamps)
                Dataset of labelled instances
        """
        self.lam = 0
        # prepare stratified sampling
        tot_samples = (DL.shape[0] + X.shape[0])*sample_size
        ratio = X.shape[0] / (X.shape[0] + DL.shape[0])
        nsamples_DL = math.ceil(tot_samples*(1-ratio))
        nsamples_X = math.ceil(tot_samples*ratio)
        for _ in range(N):
            # take samples
            Xs = X[random.sample(range(X.shape[0]), nsamples_X)]
            DLs = DL[random.sample(range(DL.shape[0]), nsamples_DL)]
            # calculate lambda for samples, take mean
            self.lam += _fast_lambda(tss = np.vstack(Xs, DLs), 
                                     pts = self.patterns["ts"].to_numpy()
                                    )/N

    def _calculate_probx(self, X, pt) -> None:
        """Calculates the value of P(X | pt)
        
        Args
        ----
            - X : (array-like) instance
            - pt : (array-like) pattern
        """
        return math.exp(-self.lam*_dis(X, pt))

    def _calculate_multinomial(self) -> None:
        # TODO optimize this if code is slow
        """For each pattern, given labels, calculates l_probas
        
        Performs multinomial MLE:
            p_i = \sum_n l_n / N
            - l_i : 1 if n = i, 0 otherwise
            - N : length of labels
        """
        self.patterns["l_probas"] = self.patters["labels"].apply(
            lambda ls : np.unique(ls, return_counts=True)[1]/len(ls)
        )

    def _calculate_uncr(self, DL, X, L, k):
        """ Find X's k-nearest neighbours LN_Ks(X)
        Estimate posterior P(y = l|X) from training set
        Use to set into summation equation.
        Multiply with quota of distances.
        """
        # to obtain X'.pt 
        # pattern_key = self.instances.get_value(<index_of_X'>, "near_pt")
        # then -->
        # X.pt = self.patterns.get_value(pattern_key, "ts")
        LN_Ks = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
        z = len(LN_Ks)
        for i in range(len(LN_Ks)):
            x_current = LN_Ks[i]
            sum += (1 / z) * self._calculate_probx(X, x_current) * self._calculate_multinomial()
        post_prob = sum

    def _calculate_uti(self, DU):
        """
        Calculate utility based on a set of questions and an unlabeled dataset.

        Args:
            DU:

        Returns:

        """