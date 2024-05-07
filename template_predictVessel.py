import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import functools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering


import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score


rand_index_score = 0



def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))


def predictor_baseline(csv_path):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # k-means with K = number of unique VIDs of set1
    K = 20 
    model = KMeans(n_clusters=K, random_state=123, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    return labels_pred


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv', 'set3noVID.csv', 'set1-2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        # labels_pred = predictor_baseline(csv_path)
        labels_pred = predictorKMeans(csv_path, False,False)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')


def evaluate():
    csv_path = './Data/set1-2.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictorKMeans(csv_path, False,False)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set1-2.csv: {rand_index_score:.4f}')


def predictorKMeans(csv_path, plot_silhouette=True, plot_elbow=True, plot_clusters=True):
    # Load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})

    # Select features
    selected_features = ['LAT', 'LON', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    
    # Standardization
    X = preprocessing.StandardScaler().fit(X).transform(X)
    
    # Optimize K using elbow method
    if plot_elbow:
        distortions = []
        K_range = range(2, 30)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=123, n_init=10)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
        # Plot the elbow
        plt.plot(K_range, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k on: ' + csv_path)
        plt.show()
    
    # K-Means with optimal K
    K = 10  # Adjust based on elbow method or other criteria
    model = KMeans(n_clusters=K, random_state=123, n_init=10).fit(X)
    
    # Predict cluster numbers of each sample
    labels_pred = model.predict(X)
    
    if plot_clusters:       
        colors = cm.nipy_spectral(labels_pred.astype(float) / K)
        plt.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = model.cluster_centers_
        # Draw white circles at cluster centers
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")


        plt.title("Clusters for: " + csv_path)
        plt.show()


    # Plot silhouette if requested
    if plot_silhouette:
        silhouette_avg = silhouette_score(X, labels_pred)
        sample_silhouette_values = silhouette_samples(X, labels_pred)
        
        y_lower = 10
        ax = plt.subplots()
        for i in range(K):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[labels_pred == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / K)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        ax.set_title("Silhouette plot for KMeans clustering for: " + csv_path)
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        plt.show()

    # visualize_clusters(csv_path)

    return labels_pred


    



if __name__=="__main__":
    get_baseline_score()
    evaluate()


# def gridSearchAggloClus(trainX, testX, trainY, testY, paramGrid, k = 10):
#     model = AgglomerativeClustering()

#     grid_search = GridSearchCV(estimator=model, param_grid=paramGrid, cv=k)
#     grid_search.fit(trainX, trainY)

#     print("Best parameters:", grid_search.best_params_)
#     best_model = grid_search.best_estimator_
#     test_accuracy = best_model.score(testX, testY)
#     print("Test set accuracy:", test_accuracy)