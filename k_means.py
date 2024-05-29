import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import os
import folium
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.tri import Triangulation
from src.Dataloader import DataLoader


class KMeansClustering:

    def __init__(self, data: DataLoader):
        self.categories = None
        self.data = data
        self.n_clusters = 5
        self.pca_data = None
        self.kmeans_pred = None
        self.centroids = None

        self.merged_triangles = None
        self.access_social_services = None
        self.Education_data = None
        self.social_data = None
        self.fertility_measures = None
        self.pop_data = None
        self.mortality_data = None
        self.social = None
        self.heath_data = None

    def dimension_reduction(self):
        pca = PCA(n_components=4, svd_solver='randomized', random_state=50)
        self.pca_data = pca.fit_transform(self.data.county_data_scaled)

    def k_means_approach(self):
        k_means = KMeans(n_clusters=self.n_clusters, random_state=1)
        k_means.fit(self.pca_data)
        self.kmeans_pred = k_means.predict(self.pca_data)
        self.centroids = k_means.cluster_centers_

    def plot_cluster_results(self):
        plt.figure(figsize=(12, 10))

        # Define the specified colors for each cluster
        cluster_colors = {1: 'cyan', 2: 'green', 3: 'red', 4: 'blue', 5: 'magenta'}

        # Define manually labeled clusters with cluster 1 and 4 interchanged
        cluster_labels = {4: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 1: 'Cluster 4', 5: 'Cluster 5'}

        # Plot data points with varying markers based on cluster membership
        for i in range(self.n_clusters):
            scatter = plt.scatter(self.pca_data[self.kmeans_pred == i, 0],
                                  self.pca_data[self.kmeans_pred == i, 1],
                                  c=cluster_colors[i + 1], s=100, alpha=0.7,
                                  edgecolors='k', label=cluster_labels[i + 1])

        # Highlight centroids
        centroids_legend = plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                                       marker='o', s=150, color='black',
                                       label='Centroids', alpha=0.8)

        # Add interactive labels for data points
        for name, coords in zip(self.data.county_names, self.pca_data):
            plt.text(coords[0], coords[1], name, fontsize=8, alpha=0.8)

        plt.xlabel('Principal Component 1', fontsize=15, fontweight="bold")
        plt.ylabel('Principal Component 2', fontsize=15, fontweight="bold")
        plt.title('K-means Clustering Results', fontsize=20, fontweight="bold")

        # Add grid lines for better readability
        plt.grid(True, linestyle='--', alpha=0.6)

        # Manually specify legend handles and labels to ensure correct order
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color=cluster_colors[i + 1], linestyle='',
                       markersize=8, alpha=0.7) for i in range(self.n_clusters)]
        legend_labels = [cluster_labels[i + 1] for i in range(self.n_clusters)]

        # Move Cluster 1 to the first position and Cluster 4 to the fourth position
        legend_labels[0], legend_labels[3] = legend_labels[3], legend_labels[0]
        legend_handles[0], legend_handles[3] = legend_handles[3], legend_handles[0]

        # Add legend including centroids
        plt.legend(legend_handles + [centroids_legend], legend_labels + ['Centroids'],
                   prop={'size': 12}, markerscale=1.0, loc='upper left')
        plt.savefig("./figs/kmeans_plot.pdf")

    def save_map_as_pdf(self, map_url, output_file):
        # Set up options to run Chrome in headless mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        # Initialize Chrome driver
        driver = webdriver.Chrome(options=chrome_options)
        # Open the map URL
        driver.get(map_url)
        # Set the window size to capture the entire map
        driver.set_window_size(1200, 1200)

        # Save the screenshot as PNG
        driver.save_screenshot(output_file + '.png')  # Add .png extension to the filename

        # Close the driver
        driver.quit()

    def cluster_kmeans_map(self):
        # Assign predicted clusters to the data
        self.data.geo_data['cluster'] = self.kmeans_pred

        # Define custom colors for clusters
        cluster_colors = ['blue', 'green', 'red', 'cyan', 'magenta']

        # Define manually labeled clusters with cluster 1 and 4 interchanged
        cluster_labels = {4: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3',
                          1: 'Cluster 4', 5: 'Cluster 5'}

        # Interchange colors for Cluster 1 and Cluster 4
        cluster_colors[0], cluster_colors[3] = cluster_colors[3], cluster_colors[0]

        # Create an empty list to store coordinates of each cluster
        clusters = []

        # Extract coordinates for each cluster and store them in the list
        for cluster_id in range(self.n_clusters):
            cluster_coords = self.data.geo_data[['lat', 'lng']][
                self.data.geo_data['cluster'] == cluster_id].values.tolist()
            clusters.append(cluster_coords)

        # Create a map centered at the given coordinates
        map_center = [-0.023559, 37.9061928]
        cluster_map = folium.Map(location=map_center, zoom_start=7)

        # Add markers for each cluster with custom color
        for cluster_id, (cluster_coords, color) in enumerate(zip(clusters,
                                                                 cluster_colors), start=1):
            label = cluster_labels.get(cluster_id,
                                       f'Cluster {cluster_id}')
            for coord in cluster_coords:
                folium.CircleMarker(coord, radius=5, color=color, fill_color=color,
                                    fill_opacity=0.6,
                                    popup=f'<b style="color:{color};">{label}</b>').add_to(
                    cluster_map)

        # Add legend with reduced height and positioned to the top left
        legend_html = '''
            <div style="position: fixed; 
                        top: 10px; left: 10px; z-index:9999; font-size:20px;
                        background-color: white;
                        ">
                <!-- Removed legend title text -->
        '''
        for cluster_id, label in cluster_labels.items():
            color = cluster_colors[cluster_id - 1]  # Adjust index to start from 0
            # Increase the marker size by adjusting the font size of the Unicode character
            legend_html += f'<p style="margin:10px;"><span style="color:{color}; ' \
                           f'font-weight:bold; ' \
                           f'font-size:30px;">&#9679;</span> <b style="color:' \
                           f'{color};">{label}</b></p>'

        legend_html += '</div>'
        cluster_map.get_root().html.add_child(folium.Element(legend_html))

        # Save the map as an HTML file
        output_file = os.path.join('../figs', 'cluster_map.html')
        cluster_map.save(output_file)

        # Convert HTML file to PDF
        pdf_output_file = os.path.join('../figs', 'cluster_map.pdf')
        self.save_map_as_pdf('file://' + os.path.abspath(output_file), pdf_output_file)

    def data_manipulation(self):
        # Normalize the data on the interval [0, 1]
        data = pd.DataFrame(self.data.data.values /
                                      np.sum(self.data.data.values, axis=0) * 100)

        # Define measures for easier reference
        categories = {
            "fertility": [9, 10, 11, 8],
            "mortality": [4, 5, 6, 7],
            "social": [13, 12, 15, 14],
            "education": [19, 16, 17, 18],
            "population": [0, 1, 3, 2],
            "access_social_services": [21, 23, 20, 22]
        }

        # Initialize empty lists to store data
        data_categories = {}
        for category, indices in categories.items():
            data_categories[category] = [data.iloc[:, [i]] for i in indices]

        # Reshape and store data
        for category, data_list in data_categories.items():
            reshaped_data = np.array(data_list).reshape(4, 47).tolist()
            setattr(self, category, reshaped_data)

        # Merge data to generate stacked plots
        triangles = []
        for i in range(4):
            triangle = []
            for category in categories.keys():
                triangle += self.__getattribute__(category)[i]
            triangles.append(triangle)

        # Store merged triangles
        self.merged_triangles = triangles
        # Initialize empty dictionary to store category data

        # Arrange categories for merging triangles
        arranged_categories = {
            'triangle_1': [categories["fertility"], categories["mortality"],
                           categories["population"]],
            'triangle_2': [categories["social"], categories["education"],
                           categories["access_social_services"]],
            'triangle_3': [categories["fertility"], categories["education"],
                           categories["access_social_services"]],
            'triangle_4': [categories["mortality"], categories["social"],
                           categories["population"]]
        }

        # Store arranged categories
        self.categories = arranged_categories

    def create_demo_data(self, M, N):
        # create some demo data for North, East, South, West
        # note that each of the 4 arrays can be either 2D (N by M) or 1D (N*M)
        # M columns and N rows
        valuesN = np.repeat(np.abs(np.sin(np.arange(N))), M)
        valuesE = np.arange(M * N) / (N * M)
        valuesS = np.random.uniform(0, 1, (N, M))
        valuesW = np.random.uniform(0, 1, (N, M))
        return [valuesN, valuesE, valuesS, valuesW]

    def triangulation_for_triheatmap(self, M, N):
        xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))
        xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        cstart = (M + 1) * (N + 1)  # indices of the centers

        trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        triangul = [Triangulation(x, y, triangles)
                    for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]

        M, N = 47, 6  # e.g. 47 counties, 6 rows
        # values = self.create_demo_data(M, N)
        values = self.merged_triangles

        cmaps = ['Greens', 'Reds', 'Blues', 'Purples']
        triangle_labels = ['Top Triangle', 'Right Triangle', 'Bottom Triangle', 'Left Triangle']

        triangle_to_categories = {
            "Top Triangle": ["Contraceptive_prevalence", "Infant_Mortality",
                             "Employment_rate", "HIV_Prevalence",
                             "Population_size", "Urbanization"],
            "Right Triangle": ["Fertility", "Under_Five_Mortality", "Crime_index",
                               "Education_level",
                               "Population_Density", "Electricity_access"],
            "Bottom Triangle": ["Birth_Rate", "Death_Rates", "Poverty_Rate", "Literacy_Rates",
                                "Growth_Rates", "Land_size"],
            "Left Triangle": ["Household_Size", "healthfacility_delivery", "Unemployment_Rate",
                              "Child_Marriage_Prevalence", "Gross_County_Product",
                              "Healthcare_Facility_Density"]
        }

        cmap_to_triangle = dict(zip(cmaps, triangle_labels))
        norms = [plt.Normalize(0, 1) for _ in range(4)]
        fig, ax = plt.subplots(figsize=(190, 80))
        all_colorbars = []

        for i, (t, val, cmap) in enumerate(zip(triangul, values, cmaps)):
            imgs = ax.tripcolor(t, np.ravel(val), cmap=cmap, ec='white')
            # Generate colorbar for each triangle plot
            start_x = 0.2  # Initial start position for the first colorbar
            cbar_ax = fig.add_axes([start_x + 0.2 * i, 0.1, 0.15, 0.05])
            cbar = plt.colorbar(imgs, cax=cbar_ax, orientation='horizontal')
            triangle_label = cmap_to_triangle[cmap]
            cbar.set_label(triangle_label, fontsize=100)
            cbar.ax.tick_params(labelsize=100)
            cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _:
                                                                '{:.0f}%'.format(x)))
            all_colorbars.append(cbar)

        # Combine all the colorbars into one horizontal colorbar
        total_width = sum([cbar.ax.get_position().width for cbar in all_colorbars[4:]])
        combined_start_x = (1 - total_width) / 2
        cbar_ax_combined = fig.add_axes([combined_start_x, 0.05, total_width, 0.01])
        cbar_combined = fig.colorbar(all_colorbars[2].mappable,
                                     cax=cbar_ax_combined, orientation='horizontal')

        ax.set_xticks(range(M), labels=self.data.county_names, rotation=90, fontsize=130)
        ax.set_yticks(range(N), labels=["Fertility measures", "Mortality measures",
                                        "Social Measures",
                                        "Education Measures", "Population & development",
                                        "Access to social services"],
                      rotation=0, fontsize=130)
        ax.invert_yaxis()
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells
        plt.tight_layout()
        plt.savefig("figs/stacked.pdf")

    def triangulation_for_triheatmap2(self, M, N):
        xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))
        xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        cstart = (M + 1) * (N + 1)  # indices of the centers

        trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        triangul = [Triangulation(x, y, triangles)
                    for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]

        M, N = 1, 6  # e.g. 47 counties, 6 rows
        values = self.categories

        triangle_labels = ['Top Triangle', 'Right Triangle', 'Bottom Triangle', 'Left Triangle']

        triangle_to_categories = {
            "Top Triangle": ["Contraceptive \n prevalence", "Infant \n Mortality \n rates",
                             "Employment \n rate",
                             "HIV \n Prevalence \n rates", "Population \n size", "Urbanization"],
            "Right Triangle": ["Fertility \n rates", "Under-Five \n Mortality \n rates",
                               "Crime \n index",
                               "Education \n level",
                               "Population \n Density", "Electricity \n access"],
            "Bottom Triangle": ["Birth \n Rate", "Death \n Rates", "Poverty \n Rate",
                                "Literacy \n Rates",
                                "Growth \n Rates", "Land \n size"],
            "Left Triangle": ["Household\n Size", "Healthcare \n facility\n delivery",
                              "Unemploy-\n ment \n Rate",
                              "Child \n Marriage\n rates", "Gross \n Domestic\n Product",
                              "Healthcare \n Facility \n Density"]
        }
        colors = {'Top Triangle': 'green', 'Right Triangle': 'red', 'Bottom Triangle': 'blue',
                  'Left Triangle': 'purple'}

        fig, ax = plt.subplots(figsize=(30, 60))

        for t, category_labels in zip(triangul, triangle_labels):
            category_names = triangle_to_categories[category_labels]
            triangle_center_x = np.mean(t.x[t.triangles], axis=1)
            triangle_center_y = np.mean(t.y[t.triangles], axis=1)
            for i, (x, y) in enumerate(zip(triangle_center_x, triangle_center_y)):
                category_name = category_names[i]
                ax.text(x, y, category_name, ha="center", va="center", fontsize=35,
                        fontweight="bold", color=colors[category_labels])

        for t in triangul:
            ax.triplot(t, color='black')  # Plot empty triangles
        ax.set_xticks(range(M))
        ax.set_xlabel(self.data.county_names[0], fontsize=80, fontweight="bold")
        ax.set_yticks(range(N), labels=["Fertility measures", "Mortality measures", "Social Measures",
                                        "Education Measures", "Population & development",
                                        "Access to social services"],
                      rotation=0, fontsize=60, fontweight="bold")
        ax.invert_yaxis()
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells

        plt.tight_layout()
        plt.savefig("figs/stacked2.pdf")







