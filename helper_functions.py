import math
from subgroup import Subgroup
import statsmodels.api as sm
from copy import deepcopy
import pandas as pd

def regression(subgroup_target, dataset_target, comparecache, use_complement=False):
    """This function returns 2 values after performing a linear regression
    The first value is the quality metric (entropy and difference in coefficient from the original model) 
    The second value is the coefficient from the new model"""
    if len(subgroup_target) < 20: # less than 20 rows is not enough to build a model on.
        return 0, None
    if len(subgroup_target.columns) != 2:
        raise ValueError("Correlation metric expects exactly 2 columns as target variables")
    x_col, y_col = list(subgroup_target.columns)
    est = sm.OLS(subgroup_target[y_col], subgroup_target[x_col])
    est = est.fit()
    coef = est.summary2().tables[1]['Coef.'][x_col]
    p = est.summary2().tables[1]['P>|t|'][x_col]
    if math.isnan(p):
        return 0, 0
    if (1 - p) < 0.99: # if the p-value is bad, don't use this model
        return 0, 0
    smallestcoefdiff = min([abs(coef - i) for i in comparecache]) 
    return entropy(subgroup_target, dataset_target) * smallestcoefdiff, coef

def entropy(subgroup_target, dataset_target):
    """Function that calculates entropy. This is used in the regression function"""
    n_c = max(1, len(dataset_target) - len(subgroup_target))
    n = len(subgroup_target)
    N = len(dataset_target)
    return -n/N * math.log(n/N) - n_c/N * math.log(n_c/N)

def create_subgroup_lists(subgroup, column: str, settings: dict):
    """This function takes a subgroup and column and makes all the possible subgroup splits on that column
    This is then returned as a list"""
    resultinggroups = []
    if column in subgroup.description:
        return []
    data = subgroup.data
    values = list(data[column].unique())
    if len(values) == 1:  # No need to make a split for a single value
        return []
    if column in settings['object_cols'] or len(values) < settings['n_bins']:
        while len(values) > 0:
            value = values.pop(0)
            subset = data[data[column] == value]
            resultinggroups.append( Subgroup(subset, deepcopy(subgroup.description).extend(column, value)))
    else:  # Float or Int
        if settings['bin_strategy'] == 'equidepth':
            _, intervals = pd.qcut(data[column].tolist(), q=min(settings['n_bins'], len(values)),
                                   duplicates='drop', retbins=True)
        else:
            raise ValueError(f"Invalid bin strategy `{settings['strategy']}`")
        intervals = list(intervals)
        lower_bound = intervals.pop(0)
        while len(intervals) > 0:
                upper_bound = intervals.pop(0)
                subset = data[(data[column] > lower_bound) & (data[column] <= upper_bound)]
                resultinggroups.append( Subgroup(subset, deepcopy(subgroup.description).extend(column, [lower_bound, upper_bound])) )
                lower_bound = upper_bound
    return resultinggroups


