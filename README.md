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
  query_strategy = select_query_strategy(alm)
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

Where classifier is defined during the definition of the ALmanager and must be ```sklearn.neighbors.KNeighborsClassifier```.

#### NOTES ABOUT THE IMPLEMENTATION
I sketched the general structure of the ACTS algorithm in the file, highlighting the necessary steps.

I based the structure of the function on the ```uncertainty_sampling``` of the library ```modAL```, adding a few things 
(DL and L as arguments).

A few problems have emerged, they have a quick solution luckilly:

* The query strategy needs two additional arguments than uncertanty sampling: the labelled data, which i called DL and their labels L.
  To make this work, the function ```va_builder.utils.alutils.select_query_strategy``` needs to be modified by adding an additional ```elif```
* The patterns must be stored even after the acts has run, since they need to be updated everytime. I gave a simple solution to the problem.
* What is a pattern I still don't know... maybe it is necessary to build an object. we'll see.

The steps of the algorithm are the following:

* Initialize/update steps
* Calculate utility for each TS
* Calculate uncertainty for each TS 
* Calculate question informativeness
* Return the most informative examples

It is not the best solution, maybe building a callable class might have been better. But this solution is a bit simpler and does not require large modifications of the code in VA_BUILDER, except for the ```select_query_strategy``` function.
