from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from collections import Counter

'''
Randomly removing data until the dataset is balanced 
'''
def random_sampling(x, y):
    print("\n", "-"*70)
    print("Under Sampling using RandomUnderSampler with strategy as all")
    print('Original dataset shape: ', Counter(y))
    us = RandomUnderSampler(sampling_strategy='all', random_state=32)

    X_res, y_res = us.fit_resample(x, y)
    print('Resampled dataset shape: ', Counter(y_res))

    return X_res, y_res

'''
A modified Condensed Nearest Neighbors(CNN).
Which finds the desired samples from the majority class that is having the lowest Euclidean distance
from the minority class then removing it.
'''
def tomek_links(x, y):
    print("\n", "-"*70)
    print("Under Sampling using TomekLinks")
    print('Original dataset shape: ', Counter(y))
    us = TomekLinks()

    X_res, y_res = us.fit_resample(x, y)
    print('Resampled dataset shape: ', Counter(y_res))

    return X_res, y_res

'''
Selects majority class samples based on their distance to minority class
'''
def near_miss(x, y):
    print("\n", "-"*70)
    print("Under Sampling using Near Miss with default values")
    print('Original dataset shape: ', Counter(y))
    us = NearMiss()

    X_res, y_res = us.fit_resample(x, y)
    print('Resampled dataset shape: ', Counter(y_res))

    return X_res, y_res