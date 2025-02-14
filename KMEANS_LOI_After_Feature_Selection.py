import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_and_preprocess_data(filepath):
    """Load dataset from a CSV file and standardize the features."""
    dataset = pd.read_csv(filepath)
    X = dataset.iloc[:, :-1].values  # Features (all columns except the last one)
    y = dataset.iloc[:, -1].values   # Target variable (last column)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize the data
    return X_scaled, y, scaler

def perform_kmeans_clustering(X, n_clusters_range):
    """Perform K-Means clustering for different values of n_clusters."""
    clustering_results = {}
    for n_clusters in n_clusters_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        clusterer.fit(X)  # Fit the model to the dataset
        cluster_labels = clusterer.predict(X)  # Predict the cluster labels
        
        # Evaluate clustering performance
        silhouette_avg = silhouette_score(X, cluster_labels)
        inertia = clusterer.inertia_

        clustering_results[n_clusters] = {
            'silhouette_score': silhouette_avg,
            'inertia': inertia
        }

        # Print clustering results
        print(
            f"For n_clusters = {n_clusters}, "
            f"Silhouette Score: {silhouette_avg}, "
            f"Inertia: {inertia}"
        )

    return clustering_results, clusterer

def perform_pca(X, n_components):
    """Perform PCA and return transformed data and PCA components."""
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    # Print PCA results
    print(f"PCA Variance: {pca.explained_variance_ratio_}")
    print(f"PCA Cumulative Variance: {pca.explained_variance_ratio_.cumsum()}")
    print(f"PCA Singular Values: {pca.singular_values_}")
    
    return X_pca, pca

def shap_analysis(X, cluster_labels, pca, feature_names):
    """Perform SHAP analysis to explain clustering and PCA results."""
    # SHAP for clustering
    print("\nPerforming SHAP analysis for clustering...")
    explainer = shap.KernelExplainer(lambda X: cluster_labels, X)  # KernelExplainer for KMeans clustering
    shap_values = explainer.shap_values(X)
    
    # Visualize SHAP clustering results
    shap.summary_plot(shap_values, X, feature_names=feature_names)
    
    # SHAP for PCA
    print("\nPerforming SHAP analysis for PCA...")
    pca_explainer = shap.Explainer(pca.transform, X)
    pca_shap_values = pca_explainer(X)
    
    # Visualize SHAP PCA results
    for i in range(pca.n_components):
        print(f"SHAP for PCA Component {i + 1}")
        shap.waterfall_plot(pca_shap_values[:, i])

def apply_clustering_and_pca_on_new_data(new_data_filepath, scaler, clusterer, pca):
    """Apply the pre-trained scaler, KMeans, and PCA to a new dataset."""
    # Load and preprocess new dataset
    new_data = pd.read_csv(new_data_filepath)
    X_new = new_data.iloc[:, :-1].values
    X_new_scaled = scaler.transform(X_new)
    
    # Predict clusters on new data
    cluster_labels_new = clusterer.predict(X_new_scaled)
    
    # Apply PCA transformation on new data
    X_new_pca = pca.transform(X_new_scaled)
    
    return cluster_labels_new, X_new_pca

def main():
    """Main function to execute the entire workflow."""
    # Filepaths for the datasets
    dataset_filepath = "XGBOOST加数据归一化_TOP85_dataset.csv"
    pubmed_filepath = "20240612_SMILES_COMBINE_48w-valid_smiles_TOP85_dataset.csv"

    # Step 1: Load and preprocess the main dataset
    X, y, scaler = load_and_preprocess_data(dataset_filepath)

    # Step 2: Perform KMeans clustering on the main dataset
    clustering_results, clusterer = perform_kmeans_clustering(X, range(5, 6))  # Example: Try clustering with 5 clusters

    # Step 3: Perform PCA on the main dataset
    X_pca, pca = perform_pca(X, n_components=38)

    # Step 4: SHAP analysis for clustering and PCA
    feature_names = pd.read_csv(dataset_filepath).columns[:-1]  # Extract feature names
    cluster_labels = clusterer.labels_
    shap_analysis(X, cluster_labels, pca, feature_names)

    # Step 5: Apply clustering and PCA to the new PubMed dataset
    apply_clustering_and_pca_on_new_data(pubmed_filepath, scaler, clusterer, pca)

if __name__ == "__main__":
    main()
