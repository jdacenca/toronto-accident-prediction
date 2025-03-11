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
