""""
This script performs initial clustering and dimensionality reduction on a dataset. It:

Loads a dataset from a CSV file.
Scales the features using standardization.
Performs KMeans clustering for a specified range of clusters and prints silhouette scores and inertia.
Performs Principal Component Analysis (PCA) to reduce dimensionality and prints explained variance, cumulative variance, and singular values.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def load_data(file_path):
    """Load dataset from a CSV file."""
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y

def scale_data(X):
    """Standardize the dataset."""
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled

def perform_kmeans(X, cluster_range):
    """Perform KMeans clustering and print silhouette scores and inertia."""
    for n_clusters in cluster_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        clusterer.fit(X)
        cluster_labels = clusterer.predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        inertia = clusterer.inertia_

        print(
            f"For n_clusters = {n_clusters}, "
            f"The average silhouette_score is: {silhouette_avg}, "
            f"The inertia is: {inertia}"
        )

def perform_pca(X, n_components):
    """Perform PCA on the dataset and print results."""
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)

    pca_variance = pca.explained_variance_ratio_
    pca_cumulative_variance = pca.explained_variance_ratio_.cumsum()
    pca_singular_values = pca.singular_values_

    print(
        f"For PCA Variance: {pca_variance}",
        f"For PCA Cumulative Variance: {pca_cumulative_variance}",
        f"The PCA Singular Values: {pca_singular_values}",
        sep="\n"
    )

    return X_pca, pca

def main():
    """Main function to execute the workflow."""
    # File path
    file_path = "EP+FR+Curing_Dataset.csv"

    # Load and scale data
    X, y = load_data(file_path)
    X_scaled = scale_data(X)

    # Perform KMeans clustering
    cluster_range = range(11, 12)
    perform_kmeans(X_scaled, cluster_range)

    # Perform PCA
    n_components = 133
    perform_pca(X_scaled, n_components)

if __name__ == "__main__":
    main()
