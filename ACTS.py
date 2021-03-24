"""
Implementation of the ACTS algorithm as a query strategy for the va_builder framework.
For any additional information see documentation.
"""
import pandas as pd
import numpy as np
import random
import math
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from pyts.classification import LearningShapelets
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
    assert n >= m, "In _dis, a sequence is longer than a pattern"
    dist_array = np.array([
        np.linalg.norm(
            X[i:(i+m)] - pt[:]
        )/m
        for i in range(0, n-m+1)
    ])
    return dist_array.min()
        

def _fast_lambda(tss : np.ndarray, pts : np.ndarray) -> float:
    """Wrapper function used in ACTS._calculate_lambda
    Calculates mean of _dis(ts, pt) for every possible ts in tss
    and pt in pts, returns mean^-1
    
    Args
    ----
        - tss : (n_instances, n_timestamps)
            2d array of instances
        - pts : (n_instances, ), dtype : object
            array of arrays
    
    Returns
    -------
        - lam : float
            Mean of _dis(ts, pt) between al ts in tss and pt in pts
    """
    mean = np.mean(
        np.array([
            [ _dis(ts, pt) for pt in pts ]
            for ts in tss
        ]).flatten()
    )
    return 1./mean


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
            
        - label_set : np.ndarray
            array of possible label values
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
        # TODO ADD CHECK_ARRAY TO SET EVERYTHING TO THE RIGHT TYPE AND DIMENSION
        if self.patterns is None:
            self._initialize_instances(DL, L, Li)
            self._initialize_patterns()
            self._assign_instances(empty_only=False)
            self.label_set = np.unique(L)
        else:
            self._update_instances(DL, L, Li)
            self._assign_instances(empty_only=True)
            self._assign_patterns()
            self._drop_empty_patterns()
            self._update_patterns_alt()
            self._assign_instances(empty_only=False)
            self._assign_patterns()
            self._drop_empty_patterns()
            
        # MODELING
        self.lam = self._calculate_lambda(X, DL)
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
            "key" : Li,
            "ts" : [DL[i] for i in range(DL.shape[0])],
            "label" : L,
            "near_pt" : [np.nan for _ in L]
        }).set_index("key")

    def _initialize_patterns(self) -> None:
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
        """For each instance, update near_pt
        
        Args : 
            empty_only : (bool) if true, only instances with n_pt = np.nan are
                updated 
        """
        if empty_only:
            indexes = self.instances[self.instances["near_pt"].isna()].index
        else:
            indexes = self.instances.index
        patterns_array = self.patterns["ts"].to_numpy(dtype="object")
        int_pattern_idx = [
            np.argmin([
                _dis(ts, pt) for pt in patterns_array 
            ]).astype("int") 
            for ts in np.stack(self.instances.loc[indexes, "ts"])
        ]    
        self.instances.loc[indexes, "near_pt"] = [
            k(pt) 
            for pt in patterns_array[int_pattern_idx]
        ]

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
            try:
                self.patterns.at[index, "inst_keys"] = np.array(
                    nn_instances.index
                )
            except ValueError:
                print(self.patterns)
                raise ValueError("THAT THING")
            self.patterns.at[index, "labels"] = nn_instances["label"].to_numpy()

    def _update_patterns(self) -> None:
        """For each pattern, check if mixed, 
            if yes, split (delete old pattern, add new ones)
        """
        aux_series = self.patterns["labels"].apply(
            lambda x : len(np.unique(x))
        )
        mixed_pts = self.patterns[aux_series>1][["inst_keys", "labels"]]
        #dropping mixed patterns
        self.patterns = self.patterns.drop(mixed_pts.index)
        # this will contain the new patterns
        for _, row in tqdm(mixed_pts.iterrows(), total=mixed_pts.shape[0]):
            instances = np.stack(self.instances.loc[row["inst_keys"], "ts"])
            labels = row["labels"]
            #generating candidates
            n_labels = len(np.unique(labels))
            clf = LearningShapelets(random_state=42, tol=0.001)
            clf.fit(X=instances, y=labels)
            shapelets_candidates = clf.shapelets_[0]
            # searching for best candidates
            best_idxs = np.empty(shape=(n_labels, ), dtype = "int8")
            for i, l in enumerate(np.unique(labels)):
                class_instances = instances[np.where(labels==l)]
                best_idxs[i] = np.argmin(
                    [
                        np.mean([
                            _dis(ts, pt) for ts in class_instances
                        ])
                        for pt in shapelets_candidates
                    ]
                )                
            best_shapelets = shapelets_candidates[best_idxs] 
            # adding best candidates to the total 
            try:
                new_pts.extend(best_shapelets)
            except UnboundLocalError:
                new_pts = [x for x in best_shapelets]
        
        # removing duplicates if there are
        keys = [k(x) for x in new_pts]
        keys_set = list(set(keys))
        if len(keys_set)!=len(keys):
            aux_dict = {key : ts for key, ts in zip(keys, new_pts)}
            new_pts = [x for x in aux_dict.values()]

        # ADD NEW PATTERNS
        self.patterns = self.patterns.append(
            pd.DataFrame({
                "key" : [k(x) for x in new_pts],
                "ts" : [x for x in new_pts],
                "inst_keys" : [np.nan for _ in new_pts], 
                "labels" : [np.nan for _ in new_pts], 
                "l_probas" : [np.nan for _ in new_pts]
            }).set_index("key")
        )

    def _update_patterns_alt(self):
        instances = np.stack(self.instances["ts"])
        labels = self.instances["label"].to_numpy()
        n_labels = len(np.unique(labels))
        clf = LearningShapelets(random_state=42, tol=0.001)
        clf.fit(X=instances, y=labels)
        shapelets_candidates = clf.shapelets_[0]
        # searching for best candidates
        best_idxs = np.empty(shape=(n_labels, ), dtype = "int8")
        for i, l in enumerate(np.unique(labels)):
            class_instances = instances[np.where(labels==l)]
            best_idxs[i] = np.argmin(
                [
                    np.mean([
                        _dis(ts, pt) for ts in class_instances
                    ])
                    for pt in shapelets_candidates
                ]
            )                
        best_shapelets = shapelets_candidates[best_idxs] 
        # adding best candidates to the total 
        try:
            new_pts.extend(best_shapelets)
        except UnboundLocalError:
            new_pts = [x for x in best_shapelets]
        
        # removing duplicates if there are
        keys = [k(x) for x in new_pts]
        keys_set = list(set(keys))
        if len(keys_set)!=len(keys):
            aux_dict = {key : ts for key, ts in zip(keys, new_pts)}
            new_pts = [x for x in aux_dict.values()]

        # ADD NEW PATTERNS
        self.patterns = pd.DataFrame({
                "key" : [k(x) for x in new_pts],
                "ts" : [x for x in new_pts],
                "inst_keys" : [np.nan for _ in new_pts], 
                "labels" : [np.nan for _ in new_pts], 
                "l_probas" : [np.nan for _ in new_pts]
            }).set_index("key")
        
            
    def _drop_empty_patterns(self) -> None:
        aux_df = pd.DataFrame(self.patterns["inst_keys"])
        aux_df["empty_bool"] = aux_df["inst_keys"].apply(lambda x : True if len(x)==0 else False)
        empty_pat_idxs = aux_df[aux_df["empty_bool"]].index
        if len(empty_pat_idxs) == 0:
            return
        self.patterns = self.patterns.drop(empty_pat_idxs)
    
    def _update_instances(self, DL, L, Li) -> None:
        """For each element in DL, check if exists in instances
            if not, add to self.istances
        
        Args : see __call__
        """
        new_keys = np.setdiff1d(Li, self.instances.index, assume_unique=True)
        new_idxs = np.where(
            np.in1d(Li, new_keys, assume_unique=True)
        )
        self.instances = self.instances.append(
            pd.DataFrame({
                "key" : new_keys,
                "ts" : [x for x in DL[new_idxs]],
                "label" : L[new_idxs],
                "near_pt" : [np.nan for _ in L[new_idxs]],
            }).set_index("key")
        )


    def _calculate_lambda(self, X : np.ndarray, DL : np.ndarray, 
                          sample_size : float = 0.01, N : int = 50) -> None:
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
            self.lam += _fast_lambda(tss = np.vstack([Xs, DLs]), 
                                     pts = self.patterns["ts"].to_numpy(dtype="object")
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
        """For each pattern, given labels, calculates l_probas
        
        Performs multinomial MLE:
            p_i = \sum_n l_n / N
            - l_i : 1 if n = i, 0 otherwise
            - N : length of labels
        """
        self.patterns["l_probas"] = self.patterns["labels"].apply(
            lambda x : np.array([
                len(np.where(x==l)[0]) for l in self.label_set
            ])/len(x)
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