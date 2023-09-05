from k_means import K_Means
from Indicators import Indicators
from Dataloader import DataLoader


class Plots:
    def __init__(self, pca_data):
        self.pca_data = pca_data
        self.data = DataLoader()

    def run(self):
        analysis = Indicators()
        analysis.PCA_apply()
        analysis.project_2D()


def main():

    # execute class indicators
    # Create plots for the paper
    ind = Indicators()
    ind.corr_data()
    ind.PCA_apply()
    ind.loadings()
    ind.corr_pcs()
    ind.dendogram_pca()
    # ind.components_project()
    # ind.project_2D()
    # ind.plot_counties()
    # ind.box_plot()

    # run the plots from k_means
    k_means = K_Means()
    k_means.dimension_reduction()
    k_means.k_means_approach()
    k_means.plot_cluster_results()
    # k_means.elbow_plot_kmeans()
    k_means.cluster_kmeans_map()

    k_means.data_manipulation()
    k_means.create_demo_data(M=47, N=6)
    k_means.triangulation_for_triheatmap(M=47, N=6)


if __name__ == "__main__":
    main()
