from nltk.cluster import util
from sklearn.cluster import k_means_
from sklearn.cluster import KMeans
import numpy as np

def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    return util.cosine_distance(X,Y)
    #return util.euclidean_distance(X,Y)

k_means_.euclidean_distances = new_euclidean_distances

def kmeans(duc_vectorized_documents, means=np.array([])):
    tags = []
    y = []
    x = np.array([])
    label = 0
    for cluster in duc_vectorized_documents:
        for document in duc_vectorized_documents[cluster]:
            y.append(label)
            tags.append(cluster + "_" + document)
            if len(x) == 0:
                x = duc_vectorized_documents[cluster][document]
            else:
                x = np.vstack([x, duc_vectorized_documents[cluster][document]])
        label += 1
    if len(means)>0:
        kmeans = KMeans(n_clusters=label,
                        init=means,
                        algorithm="full").fit(x)
    else:
        kmeans = KMeans(n_clusters=label,
                        algorithm="full").fit(x)
    return kmeans.labels_, kmeans.cluster_centers_, np.array(tags), np.array(y)
