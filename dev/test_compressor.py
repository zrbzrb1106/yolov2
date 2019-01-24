import numpy as np
import sys
import copy
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, scale, minmax_scale
from sklearn.decomposition import PCA

def compress(data):
    data_cp = data.astype(dtype=np.float)
    batch_size, width, height, channels = data_cp.shape
    data_formatted = np.reshape(data_cp, (-1, width*height, channels)).transpose((0, 2, 1))
    data_res = np.zeros(data_formatted.shape, dtype=np.float)
    batch_size, channels, length = data_formatted.shape
    flag = np.zeros(channels)
    for b in range(batch_size):
        features = data_formatted[b]
        features_copy = copy.copy(features)
        n_clusters = 8
        features_processed = minmax_scale(features, feature_range=(0, 100), axis=1)
        # pca = PCA(n_components=100)
        # pca.fit(features_normalized)
        # features_reduced = pca.transform(features_normalized)
        kmeans = KMeans(n_clusters=n_clusters, max_iter=2000, verbose=1).fit(features_processed)
        centers = kmeans.cluster_centers_
        print(np.reshape(kmeans.labels_, (8, 16)))
        print(sys.getsizeof(data), sys.getsizeof(centers) + sys.getsizeof(kmeans.labels_))
        for index, label in enumerate(kmeans.labels_):
            data_res[b][index] = data_formatted[b][index]
        data_res = np.expand_dims(data_res, axis=0).transpose((0, 1, 3, 2)).reshape((-1, 78, 78, 128))

    np.save("C:\\d\\exercise\\da\\yolov2\\dev\\data\\feature_compressed.npy", data_res)


if __name__ == "__main__":
    data = np.load('C:\\d\\exercise\\da\\yolov2\\dev\\data\\feature.npy')
    compress(data)