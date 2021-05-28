# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:09:02 2020
@author: Administrator
"""
import numpy as np
from sklearn.cluster import MeanShift
# ############################################################################# 
X = [[2920, 4920], [2320, 4320], [2720, 5120], [2000, 4400], [2320, 5520], [1360, 4560], [1920, 5920], [1520, 6320], [7040, 11040], [6880, 11680], [6880, 12320], [6640, 12640], [8520, 10520], [7920, 9920], [8320, 10720], [7920, 11120], [8720, 12720], [7520, 11520], [8560, 13360], [7120, 11920], [6720, 12320], [6520, 12520], [9400, 11400], [8800, 10800], [9200, 11600], [8480, 10880], [7840, 11040], [7200, 11200], [8000, 12800], [7600, 13200], [7400, 13400]]
X = np.reshape(X,[-1,2])
bandwidth = 4000
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)
# #############################################################################

import matplotlib.pyplot as plt
from itertools import cycle
plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(cluster_center[0]/8000, cluster_center[1]/8000, 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.plot(X[:,0]/8000,X[:,1]/8000,'r>')
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
