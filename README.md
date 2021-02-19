# ACTS 
## An Active Learning Method for Time Series Classification

A very basic implementation of the ACTS algorithm for Time Series Classification. Made for testing purposes.
The paper is available at the [following link](https://ieeexplore.ieee.org/document/7929964)

Made by [Davide Badalotti](https://github.com/Willinki) and [William Lindskog](https://github.com/WilliamLindskog) 
for MSc thesis at [Viking Analytics](https://vikinganalytics.se/).

### Structure proposal
ACTS needs to be put inside ```va_builder``` as a query strategy. 

So, given a dataset X, if there are no labels, ```va_builder``` automatically performs a ```cold_start_query_strategy``` to obtain some labels.

Then, it switches to a normal ```query_strategy```, which can be ACTS.

#### Inside ALmanager
Almanager uses the query strategy in the following way:

```python
def __init__(..., query_strategy, ...)
  self.query_strategy = query_strategy
```

Then when querying:
```python
def query(X_pool, **query_kwargs):
  ...
  query_strategy = partial(alm.query_strategy, alm.estimator)
  query_idx = query_strategy(X_pool, **query_args)
  query_idx = pool_indices[query_idx]
  self._set_query_list(query_idx)
  return query_idx
```

So we can implement ACTS as a function like:
```python
  def _function(...):
    ...
  
  def ACTS(classifier, X, n_instances, **other):
    ...
    _function()
    ...
    return indices_to_be_queried
```

Where classifier is defined during the definition of the ALmanager and must be ```sklearn.neighbors.NearestNeighbors```.
