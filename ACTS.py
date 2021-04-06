"""
Implementation of the ACTS algorithm as a query strategy for the va_builder framework.
For any additional information see documentation.
"""
from llvmlite.ir.values import Value
from numba.npyufunc import parallel
import pandas as pd
import numpy as np
import random
import math
import warnings
from sklearn.base import BaseEstimator

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from tqdm import tqdm
from pyts.transformation import ShapeletTransform
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numba import njit, prange


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
    sorted_query_idx = np.argsort(shuffled_values, kind='mergesort')[len(shuffled_values) - n_instances:]

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

    max_idx = np.argpartition(-values, n_instances - 1, axis=0)[:n_instances]
    return max_idx


def k(X: np.ndarray) -> int:
    """
    Extracts key from sequence of values.
    Used to assign a key to patterns.
    """
    return hash(
        np.mean(X)
    )


@njit(parallel=True)
def _dis(X: np.ndarray, pt: np.ndarray) -> float:
    """Given instance and pattern, calculates Dis(X, pt), sliding window.
    Used in _calculate_probx. 
    Faster
    
    Args
    ----
        - X : (array-like) instance
        - pt : (array-like) pattern
    """
    m = len(pt)
    n = len(X)
    assert n >= m, "In _dis, a sequence is longer than a pattern"
    dist_array = np.empty(shape=(n - m + 1))
    for i in prange(0, n - m + 1):
        dist_array[i] = np.linalg.norm(
            X[i:(i + m)] - pt[:]
        ) / m

    return dist_array.min()


@njit(parallel=True)
def _dis_rescaled(X: np.ndarray, pt: np.ndarray) -> float:
    """Given instance and pattern, calculates Dis(X, pt), sliding window.
    Used in _calculate_probx. 
    Rescale the data around mean.
    
    Args
    ----
        - X : (array-like) instance
        - pt : (array-like) pattern
    """
    m = len(pt)
    n = len(X)
    assert n >= m, "In _dis, a sequence is longer than a pattern"
    dist_array = np.empty(shape=(n - m + 1))
    for i in prange(0, n - m + 1):
        dist_array[i] = np.linalg.norm(
            (X[i:(i + m)] - np.mean(X[i:(i + m)])) -
            (pt[:] - np.mean(pt[:]))
        ) / m

    return dist_array.min()


def _dis_o(X: np.ndarray, pt: np.ndarray) -> float:
    """Given instance and pattern, calculates Dis(X, pt), sliding window.
    Used in _calculate_probx. 
    
    Args
    ----
        - X : (array-like) instance
        - pt : (array-like) pattern
    """
    m = len(pt)
    n = len(X)
    assert n >= m, "In _dis, a sequence is longer than a pattern"
    dist_array = np.array([
        np.linalg.norm(
            X[i:(i + m)] - pt[:]
        ) / m
        for i in range(0, n - m + 1)
    ])
    return dist_array.min()


def _dis_dtw(X: np.ndarray, pt: np.ndarray) -> float:
    """Given instance and pattern, calculates Dis(X, pt), FASTDTW.
    Used in _calculate_probx. 
    
    Args
    ----
        - X : (array-like) instance
        - pt : (array-like) pattern
    """
    m = len(pt)
    n = len(X)
    assert n >= m, "In _dis, a sequence is longer than a pattern"
    dist_array = np.array([
        fastdtw(
            X[i:(i + m)], pt[:],
            dist=euclidean
        )[0] / m
        for i in range(0, n - m + 1)
    ])
    return dist_array.min()


class ACTS:
    """Wrapper class for ACTS query strategy. 
    
    Properties
    ----------
        - patterns: pd.DataFrame
            cols : key, ts, inst_keys, idx, labels, l_probas, lambda
            
        - instances : pd.Dataframe
            cols : key, ts, label, near_pt
        
        - label_set : np.ndarray
            Contains the list of possible labels
            
        - tree : sklearn.classifier
            DecisionTree used to assign a pattern to each
            labelled instance
    """

    def __init__(self):
        self.patterns = None
        self.instances = None
        self.lam = None
        self.label_set = None
        self.transformer = ShapeletTransform(window_sizes=[0.15, 0.30, 0.05])
        self.tree = DecisionTreeClassifier(
            criterion="entropy",
            splitter="best",
            random_state=44
        )

    def __call__(self, X: np.ndarray,
                 DL: np.ndarray,
                 L: np.ndarray,
                 Li: np.ndarray,
                 n_instances: int = 1,
                 random_tie_break: bool = False,
                 **uncertainty_measure_kwargs) -> np.ndarray:
        """Sampling based on the measures defined by ACTS.
        
        Args
        ----
            - X : The pool of samples to query from.
            - DL: The instances of labelled data
                Must contain at least an instance for each label
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
            self.label_set = np.unique(L)
            self._initialize_instances(DL, L, Li)
            self._initialize_patterns()
        else:
            self._update_instances(DL, L, Li)
            self._update_patterns_alt()
            self._assign_instances()
            self._assign_patterns()
            self._drop_empty_patterns()

        # MODELING
        self._calculate_lambdas()
        self._calculate_multinomial()
        # QUESTION SELECTION
        # STEPS:
        #        compute utility
        #        compute uncertainty
        #        if uncertainty = 0, add the series to the labelled set (is it possible?)[maybe leave it as last...]
        #        calculate Q_informativeness 
        ##############################################ONLY FOR TESTING
        return self.patterns
        ##############################################

        if not random_tie_break:
            return _multi_argmax(Q_informativeness, n_instances=n_instances)

        return _shuffled_argmax(Q_informativeness, n_instances=n_instances)

    def _initialize_instances(self, DL, L, Li) -> None:
        """For each element in DL, L, Li add instance
        
        Args : see __call__
        """
        self.instances = pd.DataFrame({
            "key": Li,
            "ts": [x for x in DL],
            "label": L,
            "near_pt": [k(x) for x in DL]
        }).set_index("key")

    def _initialize_patterns(self) -> None:
        """For each instance, add pattern
        """
        # key, ts, inst_keys, labels, l_probas
        inst_values = self.instances["ts"].to_numpy()
        self.patterns = pd.DataFrame({
            "key": [k(inst) for inst in inst_values],
            "ts": [inst for inst in inst_values],
            "inst_keys": [np.array([ind]) for ind in self.instances.index],
            "labels": [np.array([lab]) for lab in self.instances["label"].to_numpy()],
        }).set_index("key")
        self.patterns["l_probas"] = self.patterns["labels"].apply(
            lambda x: np.array([
                len(np.where(x == l)[0]) for l in self.label_set
            ]) / len(x)
        )

    def _calculate_lambdas(self):
        """For each patterm calculates value of lambda. 
        """
        for index, row in self.patterns.iterrows():
            instances = np.stack(
                self.instances.loc[row["inst_keys"], "ts"]
            )
            self.patterns.at[index, "lambda"] = 1 / np.mean(
                [
                    _dis(ts, row["ts"])
                    for ts in instances
                ]
            )
            if self.patterns.at[index, "lambda"] == np.inf:
                self.patterns.at[index, "lambda"] = 0

    def _assign_instances(self) -> None:
        """For each instance, update near_pt by consulting the tree
        """
        instances_np = np.stack(self.instances["ts"])
        inst_transformed = self.transformer.transform(instances_np)
        key_inst = self.run_tree(X=inst_transformed)
        self.instances["near_pt"] = key_inst

    def _assign_patterns(self) -> None:
        """For each pattern, update inst_keys, labels.
        
        - self.patterns.inst_keys : np.array 
            Keys of instances that have the pattern as near_pt
        - self.patterns.labels : np.array 
            Keys of instances that have the pattern as near_pt
        """
        for index, _ in self.patterns.iterrows():
            nn_instances = self.instances[
                self.instances["near_pt"] == index
                ]
            self.patterns.at[index, "inst_keys"] = np.array(nn_instances.index)
            self.patterns.at[index, "labels"] = nn_instances["label"].to_numpy()

    def _update_patterns_alt(self):
        instances = np.stack(self.instances["ts"])
        labels = self.instances["label"].to_numpy()

        # transforming instances with shapelet
        inst_trmd = self.transformer.fit_transform(X=instances, y=labels)

        # extracting shapelets
        shapelets = self.transformer.shapelets_

        # training decision tree
        self.tree.fit(inst_trmd, labels)

        # ADD NEW PATTERNS
        self.patterns = (
            pd.DataFrame({
                "key": [k(x) for x in shapelets],
                "ts": [x for x in shapelets],
                "idx": [i for i in range(shapelets.shape[0])],
                "inst_keys": [np.array([]) for _ in shapelets],
                "labels": [np.array([]) for _ in shapelets],
                "l_probas": [np.array([]) for _ in shapelets]
            }).drop_duplicates('key')
                .set_index('key')
        )

    def _drop_empty_patterns(self) -> None:
        """Pretty self explanatory
        """
        aux_df = pd.DataFrame(self.patterns["inst_keys"])
        aux_df["empty_bool"] = aux_df["inst_keys"].apply(lambda x: True if len(x) == 0 else False)
        empty_pat_idxs = aux_df[aux_df["empty_bool"]].index
        self.patterns = self.patterns.drop(empty_pat_idxs)

    def _update_instances(self, DL, L, Li) -> None:
        """For each element in DL, check if exists in instances
            if not, add to self.instances
        
        Args : see __call__
        """
        new_keys = np.setdiff1d(Li, self.instances.index, assume_unique=True)
        new_idxs = np.where(
            np.in1d(Li, new_keys, assume_unique=True)
        )
        self.instances = self.instances.append(
            pd.DataFrame({
                "key": new_keys,
                "ts": [x for x in DL[new_idxs]],
                "label": L[new_idxs],
                "near_pt": [np.nan for _ in L[new_idxs]],
            }).set_index("key")
        )

    def _calculate_probx(self, X, pt) -> None:
        """Calculates the value of P(X | pt)
        
        Args
        ----
            - X : (array-like) instance
            - pt : (array-like) pattern
        """
        lam = self.patterns.at[k(pt), "lambda"]
        return math.exp(-lam * _dis(X, pt))

    def run_tree(self, X: np.ndarray):
        """Given the transformed instances in X, returns the pattern keys 
        they are assigned to.
        """
        estimator = self.tree
        # tree structure
        feature = estimator.tree_.feature
        node_indicator = estimator.decision_path(X)
        leave_id = estimator.apply(X)
        # this will contain, for each instance, the feature used 
        # by transformer in the last decision node
        patterns = np.zeros(shape=(X.shape[0], 2), dtype=int)
        pt_keys = []
        for sample_id in range(X.shape[0]):
            node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                                node_indicator.indptr[sample_id + 1]]

            patterns[sample_id, :] = sample_id, feature[node_index[-2]]
        for idx in patterns[:, 1]:
            pt_key = self.patterns[self.patterns["idx"] == idx].index
            assert len(pt_key) <= 1, "see here, pt_keys contains more than one pattern"
            pt_key = pt_key[0]
            pt_keys.append(pt_key)
        return pt_keys

    def _calculate_multinomial(self) -> None:
        """For each pattern, given labels, calculates l_probas
        
        Performs multinomial MLE:
            p_i = \sum_n l_n / N
            - l_i : 1 if n = i, 0 otherwise
            - N : length of labels
        """
        self.patterns["l_probas"] = self.patterns["labels"].apply(
            lambda x: np.array([
                len(np.where(x == l)[0]) for l in self.label_set
            ]) / len(x)
        )

    def _calculate_uncr(self, DL, X, L, k_max):
        """ Find X's k-nearest neighbours LN_Ks(X)
        Estimate posterior P(y = l|X) from training set
        Use to set into summation equation.
        Multiply with quota of distances.
        """
        # to obtain X'.pt 
        # pattern_key = self.instances.get_value(<index_of_X'>, "near_pt")
        # then -->
        # X.pt = self.patterns.get_value(pattern_key, "ts")

        # CREATE DISTANCE LIST AND MIN AND MAX VALUES
        distance_list = [_dis(X, y) for y in DL]  # ITERATES ALL ELEMENTS STORES ALL DISTANCES
        d1 = np.min(distance_list)
        d_max = np.max(distance_list)

        # RETRIEVE K NEAREST INSTANCES
        knn_idx = np.argpartition(distance_list, k_max)[:k_max] # FINDS THE INDEXES OF THE CLOSEST K-INSTANCES IN DL
        knn = DL[knn_idx] # EXTRACTS THE INSTANCES FROM DL
        


        # FOR ALL POSSIBLE LABELS, CALCULATE PROBABILITY
        prob = 0
        for i in range(len(self.patterns["l_probas"])):
            pattern_key = self.instances.get_value(Y_neighbours[i], "near_pt")
            Y_pt = self.patterns.get_value(pattern_key, "ts")
            prob += self._calculate_probx(X, Y_pt) * self.patterns["l_probas"][i]

        normalizer = np.average(prob)  # NOT SURE OF

        normalized_probability = (1 / normalizer) * prob

        return normalized_probability * np.log(normalized_probability) * (d1 / dk)

    def _calculate_uti(self, DU, DL, k_max):
        """
        Calculate utility based on a set of questions and an unlabeled dataset.
        Args:
            DU:
        Returns:
        """
        # CALCULATE NEAREST NEIGHBOURS
        near_neighbours = []
        for Y in DL:
            nn = NearestNeighbors.kneighbors(X=Y, n_neighbors=k_max, return_distance=False)
            near_neighbours.append(nn)

        # REVERSE NEIGHBOURS
        rn = []
        for X in DU:
            if X in near_neighbours:
                rn.append(X)

        # DISTANCES
        distances = []
        for i, Y in rn:
            dist = _dis_o(rn[i], Y)
            distances.append(dist)

        # CALCULATE MAX DISTANCE
        max_dist = []
        for i, Y in rn:
            maxd = max(_dis(rn[i], Y))
            max_dist.append(maxd)

        # -------
        simD = []
        for i in rn:
            sim = 1 - (distances[i] / max_dist[i])

        #
