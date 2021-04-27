# ACTS 
## An Active Learning Method for Time Series Classification

A very basic implementation of the ACTS algorithm for Time Series Classification. Made for testing purposes.
The paper is available at the [following link](https://ieeexplore.ieee.org/document/7929964)

Made by [Davide Badalotti](https://github.com/Willinki) and [William Lindskog](https://github.com/WilliamLindskog) 
for MSc thesis at [Viking Analytics](https://vikinganalytics.se/).

## Overview
The algorithm is constructed similarly to a modAL query strategy, except for some additional arguments.

It was originally built for another library, but will work with modAL as well.

More info on implementation at the documentation directory.

## Usage
To use the algorithm first clone the repository and import the object.
```{python}
from ACTS import ACTS
```

Then, create an instance of the ACTS class as:
```{python}
acts = ACTS()
```

The query strategy itself is in the `___call()___` function, so:
```{python}
query_idxs = acts(n_instances, X, DL, L, Li)
```
### Args
* `n_instances` : `int` number of instances to be queried
* `X` : `np.ndarray` of shape (n_unlabelled_data, n_points). Contains the unlabelled instances.
* `DL` : `np.ndarray` of shape (n_labelled_data, n_points). Contains all the labelled instances.
* `L` : `np.ndarray` of shape (n_labelled_data, ). Contains the labels of `DL`
* `Li` : `np.ndarray` of `dtype=int` of shape (n_labelled_data, ). Contains the indices of the labelled instances in `DL`
*
### Returns 
* `query_idxs` : `np.ndarray` of shape (n_instances, ) with the indices of the instances in `X` to be labelled
