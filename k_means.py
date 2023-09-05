import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import folium
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.tri import Triangulation
from Dataloader import DataLoader


class K_Means:

    def __init__(self):
        self.merged_triangles = None
        self.access_social_services = None
        self.Education_data = None
        self.social_data = None
        self.fertility_measures = None
        self.pop_data = None
        self.mortality_data = None
        self.social = None
        self.heath_data = None

        self.k_means_pred = None
        self.centroids = None
        self.pca_data = None
        self.pcadf = None
        self.data = DataLoader()
        self.n_cl = 5

    def dimension_reduction(self):
        pca = PCA(n_components=4, svd_solver='randomized', random_state=50)
        pca_data = pca.fit_transform(self.data.county_data_scaled)
        self.pca_data = pca_data

    def k_means_approach(self):
        # Initialize the class object
        k_means = KMeans(n_clusters=self.n_cl, random_state=1)
        k_means.fit(self.pca_data)
        # predict the labels of clusters.
        self.k_means_pred = k_means.predict(self.pca_data)
        self.centroids = k_means.cluster_centers_

    def plot_cluster_results(self):
        os.makedirs("kmeans_data", exist_ok=True)
        plt.figure(figsize=(16, 14))
        colors = ['purple', 'brown', 'green', 'red', 'orange']
        u_labels = np.unique(self.k_means_pred)
        for i in u_labels:
            plt.scatter(self.pca_data[self.k_means_pred == i, 0],
                        self.pca_data[self.k_means_pred == i, 1], label=i, c=colors[i], marker='*', s=100)

            names = list(self.data.county_names)
            for name in self.data.county_names:
                # Get the index of the name
                i = names.index(name)
                # Add the text label
                labelpad = 0.03  # Adjust this based on your dataset
                plt.text(self.pca_data[i, 0] + labelpad, self.pca_data[i, 1] + labelpad, name, fontsize=9)

            plt.legend(title="Clusters")
            plt.savefig("./kmeans_data/kmeans_plot.pdf")

    def cluster_kmeans_map(self):
        self.data.geo_data['y'] = self.k_means_pred
        # plt.scatter(self.data.geo_data['lat'], self.data.geo_data['lng'], c=self.data.geo_data['y'])
        # plt.show()

        cluster1 = self.data.geo_data[['lat', 'lng']][self.data.geo_data['y'] == 0].values.tolist()
        cluster2 = self.data.geo_data[['lat', 'lng']][self.data.geo_data['y'] == 1].values.tolist()
        cluster3 = self.data.geo_data[['lat', 'lng']][self.data.geo_data['y'] == 2].values.tolist()
        cluster4 = self.data.geo_data[['lat', 'lng']][self.data.geo_data['y'] == 3].values.tolist()
        cluster5 = self.data.geo_data[['lat', 'lng']][self.data.geo_data['y'] == 4].values.tolist()

        # Map = folium.Map(location=[0.0236, 37.9062], zoom_start=10, tiles='OpenStreetMap')
        center = [-0.023559, 37.9061928]

        Map = folium.Map(location=center, zoom_start=8)

        for i in cluster1:
            folium.CircleMarker(i, radius=2, color="blue", fill_color="lightblue").add_to(Map)
        for i in cluster2:
            folium.CircleMarker(i, radius=2, color="red", fill_color="lightred").add_to(Map)
        for i in cluster3:
            folium.CircleMarker(i, radius=2, color="green", fill_color="lightgreen").add_to(Map)
        for i in cluster4:
            folium.CircleMarker(i, radius=2, color="Sienna", fill_color="lightblue").add_to(Map)
        for i in cluster5:
            folium.CircleMarker(i, radius=2, color="navy", fill_color="lightblue").add_to(Map)

        map_kenya = folium.Map(location=center, zoom_start=8)
        for index, self.data.geo_data in self.data.geo_data.iterrows():

            location = [self.data.geo_data['lat'], self.data.geo_data['lng']]
            folium.Marker(location, popup=f'Name:{self.data.geo_data["county"]}\n y($):'
                                          f'{self.data.geo_data["y"]}').add_to(map_kenya)

        print(Map)

    def elbow_plot_kmeans(self):
        os.makedirs("kmeans_data", exist_ok=True)
        wcss = []
        for i in range(1, 20):
            model = KMeans(n_clusters=i, max_iter=300)
            model.fit(self.pca_data)
            # check the model accuracy
            model_accuracy = model.inertia_
            wcss.append(model_accuracy)
        plt.plot(range(1, 20), wcss)
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        plt.title("Elbow Method")
        plt.savefig("./kmeans_data/elbow_plot.pdf")

    def plot_cluster_centroids(self):
        u_labels = np.unique(self.k_means_pred)
        for i in u_labels:
            plt.scatter(self.pca_data[self.k_means_pred == i, 0],
                        self.pca_data[self.k_means_pred == i, 1], label=i)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=80, marker='*', color='g')
        plt.legend()
        plt.show()

    def data_manipulation(self):
        # normalize the data on the interval [0, 1]
        self.data.data = pd.DataFrame(self.data.data.values / np.sum(self.data.data.values, axis=0))
        #  Fertility measures
        contraceptive = self.data.data.iloc[:, 9:10]
        fertility_rate = self.data.data.iloc[:, 10:11]
        birth_rate = self.data.data.iloc[:, 11:12]
        household_size = self.data.data.iloc[:, 8:9]
        fertility = np.array([contraceptive, fertility_rate, birth_rate, household_size]).reshape(4, 47)
        fertility = list(fertility)
        self.fertility_measures = fertility

        # Mortality measures
        infant_mortality = self.data.data.iloc[:, 4:5]
        under_five_mortality = self.data.data.iloc[:, 5:6]
        death_rates = self.data.data.iloc[:, 6:7]
        heath_facility_delivery = self.data.data.iloc[:, 7:8]
        mortality = np.array([infant_mortality, under_five_mortality, death_rates,
                              heath_facility_delivery]).reshape(4, 47)
        mortality = list(mortality)
        self.mortality_data = mortality

        # social measures
        employment_rate = self.data.data.iloc[:, 13:14]
        crime_index = self.data.data.iloc[:, 12:13]
        poverty_rate = self.data.data.iloc[:, 15:16]
        unemployment_rate = self.data.data.iloc[:, 14:15]
        social = np.array([employment_rate, crime_index, poverty_rate, unemployment_rate]).reshape(4, 47)
        social = list(social)
        self.social_data = social

        # Education measures
        HIV_prevalence = self.data.data.iloc[:, 19:20]
        education_level = self.data.data.iloc[:, 16:17]
        literacy_rates = self.data.data.iloc[:, 17:18]
        child_marriage = self.data.data.iloc[:, 18:19]
        Education = np.array([HIV_prevalence, education_level, literacy_rates,
                              child_marriage]).reshape(4, 47)
        Education = list(Education)
        self.Education_data = Education

        # population & development
        population = self.data.data.iloc[:, 0:1]
        pop_density = self.data.data.iloc[:, 1:2]
        GDP = self.data.data.iloc[:, 3:4]
        growth_rates = self.data.data.iloc[:, 2:3]
        pop = np.array([population, pop_density, GDP, growth_rates]).reshape(4, 47)
        pop = list(pop)
        self.pop_data = pop

        # Access to social services
        urbanization = self.data.data.iloc[:, 21:22]
        electricity_access = self.data.data.iloc[:, 23:24]
        land_size = self.data.data.iloc[:, 20:21]
        health_facility_density = self.data.data.iloc[:, 22:23]
        access_social_services = np.array([urbanization, electricity_access, land_size,
                                          health_facility_density]).reshape(4, 47)
        access_social_services = list(access_social_services)
        self.access_social_services = access_social_services

        # merge the data to generate stacked plots
        # first triangle shape
        f1 = np.append(fertility[0], mortality[0])
        f2 = np.append(f1, social[0])
        f3 = np.append(f2, Education[0])
        f4 = np.append(f3, pop[0])
        f5 = np.append(f4, access_social_services[0])

        # second triangle shape
        s1 = np.append(fertility[1], mortality[1])
        s2 = np.append(s1, social[1])
        s3 = np.append(s2, Education[1])
        s4 = np.append(s3, pop[1])
        s5 = np.append(s4, access_social_services[1])

        # third triangle shape
        third = np.array([self.fertility_measures[2], self.mortality_data[2], self.social_data[2],
                       self.Education_data[2], self.pop_data[2], self.access_social_services[2]])

        # fourth triangle shape
        fourth = np.array([self.fertility_measures[3], self.mortality_data[3], self.social_data[3],
                       self.Education_data[3], self.pop_data[3], self.access_social_services[3]])

        # merge the four triangles
        self.merged_triangles = [f5, s5, third, fourth]

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
        xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
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

        cmaps = ['Greens',  'gist_earth_r', 'Set3_r', 'Paired_r']  # ['winter', 'spring', 'summer', 'autumn']
        norms = [plt.Normalize(0, 1) for _ in range(4)]
        fig, ax = plt.subplots(figsize=(190, 48))
        imgs = [ax.tripcolor(t, np.ravel(val), cmap=cmap, ec='white')
                for t, val, cmap in zip(triangul, values, cmaps)]

        ax.set_xticks(range(M), labels=self.data.county_names, rotation=90, fontsize=130)
        ax.set_yticks(range(N), labels=["Fertility measures", "Mortality measures", "Social Measures",
                                        "Education Measures", "Population & development",
                                        "Access to social services"],
                      rotation=0, fontsize=130)
        ax.invert_yaxis()
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells
        plt.tight_layout()
        plt.savefig("stacked.pdf")




