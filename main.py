from src.k_means import KMeansClustering
from src.hierarchical_clustering import Indicators
from src.Dataloader import DataLoader


def main():
    # Execute class Indicators
    data = DataLoader()
    ind = Indicators(data=data)

    # plot correlation of the socioeconomic and health indicators
    ind.corr_data()

    # visualize pca results
    ind.PCA_apply()

    # plot factor loadings for the reduced components
    ind.loadings()
    ind.corr_pcs()

    # plot the dendogram for hierarchical clustering
    ind.dendogram_pca()

    # Run the plots from KMeansClustering
    # initialize the clustering method
    k_means = KMeansClustering(data)
    k_means.dimension_reduction()
    k_means.k_means_approach()

    # plot the clusters
    k_means.plot_cluster_results()

    # visualize the clusters on a geographical map
    k_means.cluster_kmeans_map()

    # plot the indicators in small triangles
    k_means.data_manipulation()
    k_means.create_demo_data(M=47, N=6)
    k_means.triangulation_for_triheatmap(M=47, N=6)

    # plot a table that explains the triangles
    k_means.triangulation_for_triheatmap2(M=1, N=6)


if __name__ == "__main__":
    main()
