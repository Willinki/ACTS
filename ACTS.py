"""
Implementation of the ACTS algorithm as a query strategy for the va_builder framework.
For any additional information see documentation.
"""
import pandas as pd
import numpy as np
import math
import warnings
from scipy.spatial.distance import jensenshannon

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
    if n_instances <= values.shape[0]:
        print("Too few instances, returning all remaining"
              "unlabelled data")
        return values

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

        - transformer : pyts.transformation.ShapeletTransform 
            Used to compute pattern candidates. 

        - tree : sklearn.classifier
            DecisionTree used to assign a pattern to each
            labelled instance
    """

    def __init__(self):
        self.__name__ = "ACTS"
        self.patterns = None
        self.instances = None
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
            - L: The labels of DL
            - Li: Indices of labelled instances
            - n_instances: Number of samples to be returned.
            - random_tie_break: If True, shuffles utility scores to randomize the order. This
                can be used to break the tie when the highest utility score is not unique.
            - **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
                measure function.

        Returns
        -------
            query_idxs = np.ndarray of shape (n_instances, )
                The indices of the instances from X chosen to be labelled;
        """
        # MAINTAINING PATTERNS
        if self.patterns is None:
            self.label_set = np.unique(L)
            self._initialize_instances(DL, L, Li)
            self._initialize_patterns()
        self._update_instances(DL, L, Li)
        self._update_patterns_alt()
        self._assign_instances()
        self._assign_patterns()
        self._drop_empty_patterns()

        # MODELING
        self._calculate_lambdas()
        self._calculate_multinomial()
        utility = self._calculate_uti(X, DL, 16)
        uncertainty = [self._calculate_uncr(DL, x, L, 16) for x in X]
        Q_informativeness = utility + uncertainty

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
                self.patterns.at[index, "lambda"] = 100000000000

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
        """Create new patterns from the instances
        """
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
        """Pretty self-explanatory
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
        """
        Given a unlabeled instance, this function calculates the uncertainty of a label.
        Args:
            DL: Labeled data set
            X: The unlabeled time series instance
            L: Label set
            k_max: Neighbors to the unlabeled instance that are to be calculated

        Returns: The uncertainty of a label for instance X

        """

        # (1) CREATE DISTANCE LIST AND MIN AND MAX VALUES
        distance_list = [_dis(X, y) for y in
                         np.stack(self.instances["ts"])]  # ITERATES ALL ELEMENTS STORES ALL DISTANCES
        if k_max >= len(distance_list):
            k_max = len(distance_list) - 1
        knn_idx = np.argpartition(distance_list, k_max)[:k_max]  # FINDS THE INDEXES OF THE CLOSEST K-INSTANCES IN DL
        knn_keys = self.instances.iloc[
            knn_idx].index  # OBTAINS THE KEYS OF THE KNNs (key meaning the value of self.instances.key for each knn)
        # knn = DL[knn_idx]  # EXTRACTS THE INSTANCES FROM DL

        distance_list = np.sort(distance_list)
        distance_list = distance_list[:k_max]
        d1 = min(distance_list)
        d_max = max(distance_list)

        # (2) ITERATE OVER ALL POSSIBLE LABELS
        probability_list = []
        for l in self.label_set:  # UNIQUE VALUES
            sum_probability = 0
            for key in knn_keys:
                pt_key = self.instances.at[key, "near_pt"]
                pt = self.patterns.at[pt_key, "ts"]
                sum_probability += self._calculate_probx(X, pt) * self.patterns.at[pt_key, "l_probas"][l-1]
            probability_list.append(sum_probability)

        # (3) NORMALIZE AND RETURN
        norm_Z = sum(probability_list)
        uncertainty = 0
        for i in range(len(probability_list)):
            uncertainty += (probability_list[i] / norm_Z) * np.log(probability_list[i] / norm_Z) * (d1 / d_max)

        return uncertainty

    def _calculate_uti(self, DU, DL, k_max):
        """
        Calculate utility based on a set of questions and an unlabeled dataset.
        Args:
            DU: Unlabeled time series data set
            DL: Labeled time series data set
            k_max: Number of neighbors that are to be calculated

        Returns: The utility of a new labeled instance

        """
        # CONTINUE HERE ...
        # (1) CALCULATE REVERSE NEAREST NEIGHBOURS
        new_dict, X_list = self._calc_rnn(DU=DU, DL=DL, k_nn=k_max)

        # (2) FOR EACH KEY IN DICTIONARY, CALCULATED SIM_D
        key_list = new_dict.keys()
        simD = []
        for i, key in enumerate(key_list):
            X = X_list[i]
            y_values = new_dict.get(key)
            simD_part = self._sim_D(X=X, Y_values=y_values)
            simD.append(simD_part)

        simD = sum(simD)

        # (3) CALCULATE NN OF Xi THAT ARE IN DL (PERHAPS MAKE INTO A FUNCTION)
        sum_probabilities = []
        norms = []
        sum_index = 0
        for i, key in enumerate(key_list):
            X = X_list[i]
            y_values = new_dict.get(key)
            sum_index, Z = self._prob_pattern(X=X, Y_values=y_values)
            sum_probabilities.append(sum_index)
            norms.append(Z)

        prob_X = []
        for i in range(len(sum_probabilities)):
            prob_Xi = sum_probabilities[i] / norms[i]
            prob_X.append(prob_Xi)

        # DO THE SAME FOR Y
        Y_kNNs, Y_list = self._get_Y_knn(DL, k_max)

        sum_probabilities = []
        norms = []
        sum_index = 0
        for i, Y in enumerate(Y_list):
            y_values = Y_kNNs[i]
            sum_index, Z = self._prob_pattern(X=Y, Y_values=y_values)
            sum_probabilities.append(sum_index)
            norms.append(Z)

        prob_Y = []
        for i in range(len(sum_probabilities)):
            prob_Yi = sum_probabilities[i] / norms[i]
            prob_Y.append(prob_Yi)

        if len(prob_X) < len(prob_Y):
            prob_Y = prob_Y[:len(prob_X)]
        elif len(prob_X) > len(prob_Y):
            prob_X = prob_X[:len(prob_Y)]
        else:
            pass

        # (5) CALCULATE EVALUATION OF THE SIMILARITY OF X AND Y's DISTRIBUTION OVER PATTERNS
        simP = 1 - jensenshannon(prob_X, prob_Y)

        # (6) CALCULATE SIMILARITY MEASURE AND UTILITY
        # sim = []
        # for i in simP:
            # sim_part = simD[i] * simP[i]
            # sim.append(sim_part)

        utility = simD * simP

        # utility = sum(sim)
        return utility

    def _calc_rnn(self, DU, DL, k_nn):
        """
        Calculates a set of reverse nearest neighbors.
        Args:
            DU: Unlabeled time series data set
            DL: Labeled time series data set
            k_nn: k-Neighbors

        Returns: A dictionary of reverse nearest neighbors where key X is linked with a list of Y values

        """
        # GET ALL KNNs IN DU FOR Y IN DL
        dictionary = {}
        X_list = []
        for i, X in enumerate(DU):
            rnn = []
            for Y in DL:
                dist_list = [_dis(xi, Y) for xi in DU]
                if k_nn >= len(dist_list):
                    k_nn = len(dist_list) - 1
                rnn_idx = np.argpartition(dist_list, k_nn)[:k_nn]  # FINDS THE INDEXES OF THE CLOSEST K-INSTANCES IN DU
                if X in DU[rnn_idx]:
                    rnn.append(Y)
                    key = i
                    X_list.append(X)
                    dictionary[key] = rnn

        return dictionary, X_list

    def _sim_D(self, X, Y_values):
        """
        Calculates normalized distance between X and Ys
        Args:
            X: The instance that requires labelling
            dictionary: Where Y values are stored. Use X as key

        Returns: normalized distance

        """
        # (1) GET Y VALUES FROM DICTIONARY
        # values_Y = dictionary.get(X, default="Key does not exist.")

        # (2) CALCULATE DISTANCE FOR EACH Y TO X, MAX DISTANCE, AND NORMALIZED DISTANCE
        dist_list = [_dis(X, Y) for Y in Y_values]
        max_dist = max(dist_list)
        simD = [(1 - (dist / max_dist)) for dist in dist_list]
        return sum(simD)

    def _prob_pattern(self, X, Y_values):
        """
        Calculate specified quantity for each possible pattern.
        Args:
            X: Unlabeled time series data
            dictionary: Containing NN labeled time series to X

        Returns: Sum of probabilities for each pattern and normalization constant

        """
        # (1) GET Y VALUES FROM DICTIONARY
        # values_Y = dictionary.get(X, default="Key does not exist.")

        # (2) CALCULATE SUM FOR EACH PATTERN
        pattern_sums = []
        for pt in self.patterns["ts"].to_numpy():
            sum_prob = 0
            for Y in Y_values:
                I_binary = 0
                inst_key = self.get_inst_key(Y)
                pt_key = self.instances.at[inst_key, "near_pt"]
                # CORRECT
                Y_pt = self.patterns.at[pt_key, "ts"]
                if np.array_equal(Y_pt, pt):
                    I_binary = 1
                sum_prob += self._calculate_probx(X, Y_pt) * I_binary
            pattern_sums.append(sum_prob)

        # (3) NORMALIZING CONSTANT
        norm_Z = sum(pattern_sums)

        return sum(pattern_sums), norm_Z

    def _get_Y_knn(self, DL, k_nn):
        # (1) CALCULATE DISTANCES
        Y_kNNs_list = []
        Y_list_use = []
        for Y in DL:
            dist_list = [_dis(Y, y) for y in DL]  # PERHAPS MUST CHANGE SO THAT IT DOES
            # NOT CALCULATE Y TO Y
            if k_nn >= len(dist_list):
                k_nn = len(dist_list) - 2
            knn_idx = np.argpartition(dist_list, k_nn + 1)[1:k_nn + 1]
            Y_kNNs_list.append(DL[knn_idx])
            Y_list_use.append(Y)

        return Y_kNNs_list, Y_list_use

    def get_inst_key(self, Y: np.ndarray):
        """
        Given an instance (array), return its key
        """
        idx = np.where([
            np.array_equal(x, Y)
            for x in np.stack(self.instances["ts"])
        ])
        return self.instances.iloc[idx].index[0]
