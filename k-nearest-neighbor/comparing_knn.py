#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 06:50:22 2020

@author: rekib
"""

import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearnest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K value is small')
    distances = []    
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    
#    print(vote_result, confidence)
    
    return vote_result, confidence

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

accuracies = []
for i in range(25):
    random.shuffle(full_data)
    
    test_size = 0.25
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]
    
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    
    correct = 0
    total = 0
        
    
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearnest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    
#    print('Accuracy:', correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))
    
        
      
      

      
      
      
      
      
      