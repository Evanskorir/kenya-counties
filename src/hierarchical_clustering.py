import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from src.Dataloader import DataLoader


class Indicators:
    def __init__(self, data: DataLoader):
        self.data = data
        self.pca_data = np.array([])
        self.pca2 = []

    def corr_data(self):
        if not os.path.exists("figs"):
            os.makedirs("figs")
        plt.figure(figsize=(18, 18))
        sns.heatmap(self.data.data.iloc[:, 1:].corr(), cmap="rainbow")
        plt.savefig("figs/corr.pdf")

    def PCA_apply(self):
        if not os.path.exists("figs"):
            os.makedirs("figs")

        pca = PCA(svd_solver='randomized', random_state=50)
        pca.fit(self.data.county_data_scaled)

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, color='skyblue',
                edgecolor='black')
        plt.xlabel("Principal Components", fontweight='bold', fontsize=12)
        plt.ylabel("Variance Ratio", fontweight='bold', fontsize=12)
        plt.title("Variance Ratio Explained by Principal Components",
                  fontweight='bold', fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig("figs/variance.pdf")

        plt.figure(figsize=(10, 6))
        cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cum_var_ratio) + 1), cum_var_ratio, color='blue', linewidth=2)

        # Annotate cumulative explained variance for PC1, PC2, PC3,
        # and PC4 with different colored dotted lines
        colors = ['red', 'green', 'orange', 'purple']
        for i in range(4):
            plt.plot([0, i + 1], [cum_var_ratio[i], cum_var_ratio[i]],
                     linestyle='--', color=colors[i],
                     linewidth=1.5)  # Extend line to horizontal axis
            plt.scatter(i + 1, cum_var_ratio[i], color=colors[i], marker='o', zorder=5)
            plt.annotate(f'PC{i + 1}: {int(cum_var_ratio[i] * 100)}%', (i + 1, cum_var_ratio[i]),
                         textcoords="offset points",
                         xytext=(-20, 10), ha='center', fontsize=10)
            plt.plot([i + 1, i + 1], [0, cum_var_ratio[i]], color=colors[i], linestyle='--',
                     linewidth=1.5)  # Extend line to vertical axis

        # Add legend with purple color
        plt.hlines(y=cum_var_ratio[3], xmin=0, xmax=4, colors="purple", linestyles="--",
                   label='Explained Variance by 4 PCs = 74%',
                   linewidth=2)
        plt.legend(fontsize=10)
        plt.xlabel('Number of Principal Components', fontweight='bold', fontsize=12)
        plt.ylabel('Cumulative Explained Variance', fontweight='bold', fontsize=12)
        plt.title('Cumulative Explained Variance by Principal Components',
                  fontweight='bold', fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(axis='both', linestyle='--', alpha=0.5)
        plt.savefig("figs/exp_variance.pdf")

        pca2 = PCA(n_components=4, svd_solver='randomized', random_state=50)
        pca_data = pca2.fit_transform(self.data.county_data_scaled)

        self.pca_data = pca_data
        self.pca2 = pca2

        cumulative_var_explained = np.cumsum(pca2.explained_variance_ratio_)
        print("\n Cumulative variance explained by PCs:")
        for i, var in enumerate(cumulative_var_explained, start=1):
            print(f"PC{i}: {var}")

        print("\n Explained variance explained by PCs:")
        for i, var in enumerate(pca2.explained_variance_ratio_, start=1):
            print(f"PC{i}: {var}")

    def dendogram_pca(self):
        labels = list(self.data.county_names)
        colors = ['blue', 'green', 'red', 'cyan', 'magenta']

        fig, axes = plt.subplots(1, 1, figsize=(15, 12))
        sch.set_link_color_palette(colors)
        dendrogram = sch.dendrogram(sch.linkage(self.pca_data, method="ward"), color_threshold=9,
                                    get_leaves=True, leaf_rotation=90, leaf_font_size=12,
                                    show_leaf_counts=True, orientation="top", distance_sort=True,
                                    labels=labels, above_threshold_color='black', ax=axes)

        plt.title('Hierarchical Clustering Dendrogram', fontsize=25, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=20, fontweight="bold")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=15)
        axes.tick_params(axis='both', which='major', labelsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.axhline(y=200, color='gray', linestyle='--', linewidth=1)
        plt.text(-20, 200, 'Threshold = 200', fontsize=10, color='gray')

        for i, color in enumerate(colors):
            plt.plot([], [], color=color, label=f'Cluster {i + 1}', linewidth=5, alpha=0.8)
        plt.legend(loc='upper right', fontsize=15)

        dendrogram_color = 'lightgray'
        plt.axhspan(0, 200, facecolor=dendrogram_color, alpha=0.2)

        plt.tight_layout()
        plt.savefig("figs/Dendrogram.pdf", dpi=300)

    def corr_pcs(self):
        if not os.path.exists("figs"):
            os.makedirs("figs")

        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        im = ax.imshow(self.pca2.components_, cmap='coolwarm', alpha=0.9,
                       interpolation="nearest")

        cbar = plt.colorbar(im, orientation='horizontal', pad=0.45, shrink=0.5)
        cbar.ax.tick_params(labelsize=20)
        plt.xticks(ticks=np.arange(len(pd.DataFrame(self.data.data).columns)),
                   labels=pd.DataFrame(self.data.data).columns, fontsize=20,
                   fontdict={'weight': 'bold'}, rotation=90)
        plt.yticks(ticks=np.arange(0, 4),
                   labels=['PC1', 'PC2', 'PC3', 'PC4'], fontsize=18)

        # plt.title('Principal Component Analysis', fontsize=25, fontdict={'weight': 'bold'})
        # Add annotations
        for i in range(len(self.pca2.components_)):
            for j in range(len(self.pca2.components_[i])):
                plt.text(j, i, f'{self.pca2.components_[i, j]:.2f}', ha='center',
                         va='center', color='black',
                         fontsize=12)

        plt.tight_layout()
        plt.savefig("figs/components.pdf")

    def loadings(self):
        if not os.path.exists("figs"):
            os.makedirs("figs")
        pcs_com = pd.DataFrame({'PC1': self.pca2.components_[0], 'PC2': self.pca2.components_[1],
                                'PC3': self.pca2.components_[2], 'PC4': self.pca2.components_[3]})
        pcs_com['Features'] = self.data.data.columns.values
        loadings = pcs_com.set_index('Features')
        plt.figure(figsize=(18, 18))
        ax = sns.heatmap(loadings, cmap='rainbow')
        plt.savefig("figs/loadings.pdf")

    def components_project(self):
        plt.figure(figsize=(14, 12))
        plt.scatter(self.pca2.components_[0], self.pca2.components_[1])
        plt.xlabel('Principal Component 1 (36.5%)')
        plt.ylabel('Principal Component 2 (16.3%)')
        for i, txt in enumerate(self.data.data.columns):
            plt.annotate(txt, (self.pca2.components_[0][i], self.pca2.components_[1][i]))
        plt.tight_layout()
        plt.savefig("figs/projection.pdf")

    def project_2D(self):
        pca3 = PCA(n_components=2, svd_solver='randomized', random_state=42)
        pca_dataa = pca3.fit_transform(self.data.county_data_scaled)
        components = pd.DataFrame(pca3.components_.T, index=pd.DataFrame(
            self.data.data).columns,
                                  columns=['PCA1', 'PCA2'])

        # plot size
        plt.figure(figsize=(16, 12))

        # main scatter-plot
        plt.scatter(pca_dataa[:, 0], pca_dataa[:, 1], edgecolors='black',
                    alpha=0.4, s=20)
        print(pca_dataa.shape)
        plt.xlabel('First Dim (36.5%)')
        plt.ylabel('Second Dim (16.3%)')
        plt.ylim(10, -10)
        plt.xlim(10, -10)

        # individual feature values
        ax2 = plt.twinx().twiny()
        ax2.set_ylim(-0.4, 0.4)
        ax2.set_xlim(-0.4, 0.4)

        # reference lines
        ax2.hlines(0, -0.4, 0.4, linestyles='dotted', colors='black')
        ax2.vlines(0, -0.4, 0.4, linestyles='dotted', colors='black')

        # offset for labels
        offset = 1.06

        # arrow & text
        for a, i in enumerate(components.index):
            ax2.arrow(0, 0, components['PCA1'][a], -components['PCA2'][a],
                      alpha=0.2, facecolor='black', head_width=0.005)
            ax2.annotate(i, (components['PCA1'][a] * offset,
                             -components['PCA2'][a] * offset), color='black')
        plt.savefig("figs/2D projection.pdf")

    def plot_counties(self):
        # Params
        n_samples = 47  # number of countries
        m_features = 25  # number of economic indicators
        selected_names = list(self.data.county_names)

        # Generate
        np.random.seed(42)
        names = list(self.data.county_names)
        labels = [np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                                    'L', 'M', 'N',
                                    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                                    'Z', '1', '2', '3', '4', '5', '6']) for i in range(n_samples)]
        features = np.random.random((n_samples, m_features))

        # Label to color dict (manual)
        label_color_dict = {'A': 'red', 'B': 'peru', 'C': 'blue', 'D': 'magenta',
                            'E': 'orange', 'F': 'yellow', 'G': 'black', 'H': "plum", 'I': 'olive',
                            'J': 'green', 'K': 'Plum', 'L': 'tomato', 'M': 'lime',
                            'N': 'YellowGreen', 'O': 'LemonChiffon', 'P': 'DarkGoldenrod',
                            'Q': 'Maroon', 'R': 'pink', 'S': 'ForestGreen', 'T': 'Sienna',
                            'U': 'chocolate',
                            'V': 'brown', 'W': 'DeepPink', 'X': 'DarkOrchid', 'Y': 'Violet',
                            'Z': 'MediumBlue',
                            '1': 'Aquamarine', '2': 'BlueViolet', '3': 'purple', '4': 'navy',
                            '5': 'aqua',
                            '6': 'teal', '7': 'dimgray', '8': 'dimgrey', '9': 'snow', '10':
                                'tan',
                            '11': 'khaki', '12': 'wheat', '13': 'bisque', '14': 'coral',
                            '15': 'limegreen'
                            }

        # Color vector creation
        col = [label_color_dict[label] for label in labels]

        # Create the scatter plot
        plt.figure(figsize=(16, 14))
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1],
                    c=col, alpha=0.5)

        # Add the labels
        for name in self.data.county_names:
            # Get the index of the name
            i = names.index(name)

            # Add the text label
            labelpad = 0.03  # Adjust this based on your dataset
            plt.text(self.pca_data[i, 0] + labelpad, self.pca_data[i, 1] +
                     labelpad, name, fontsize=9)

            # Mark the labeled observations with a star marker
            plt.scatter(self.pca_data[i, 0], self.pca_data[i, 1],  c=col[i],
                        vmin=min(col), vmax=max(col), marker='*', s=100)

        # Add the axis labels
        plt.xlabel('First Dim (36.5%)')
        plt.ylabel('Second Dim (16.3%)')
        plt.savefig("figs/counties projection.pdf")

    def box_plot(self):
        fig, axs = plt.subplots(5, 5, figsize=(19, 17))
        # Under-Five Mortality Rate: is the death of young children under the age of 5 per 1000 live births.
        UMR = self.data.data2[["County", 'Under_Five_Mortality']].sort_values(
            'Under_Five_Mortality', ascending=False).head(7)
        ax = sns.barplot(x='County', y='Under_Five_Mortality', data=UMR, ax=axs[0, 0])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='', ylabel='Child Mortality Rate')
        ax.set(xlabel='Counties with high Under-Five Mortality Rate',
               ylabel='Under-Five Mortality Rate')

        # Infant Mortality Rate: is the death of young children under the age of 1.
        IMR = self.data.data2[['County', 'Infant_Mortality']].sort_values('Infant_Mortality',
                                                                          ascending=False).head(7)
        ax = sns.barplot(x='County', y='Infant_Mortality', data=IMR, ax=axs[0, 1])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high Infant Mortality Rates', ylabel='Infant Mortality Rates')

        # Crude Birth Rates: number of live births occurring during the year,
        # per 1,000 population estimated at midyear
        CBR = self.data.data2[['County', 'Birth_Rate']].sort_values('Birth_Rate',
                                                                    ascending=False).head(7)
        ax = sns.barplot(x='County', y='Birth_Rate', data=CBR, ax=axs[0, 2])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high Birth Rates', ylabel='Crude Birth Rates')

        # Crude Death Rates: the number of deaths in a given period
        # divided by the population exposed to risk of
        # death in that period.
        CDR = self.data.data2[['County', 'Death_Rates']].sort_values('Death_Rates',
                                                                     ascending=False).head(7)
        ax = sns.barplot(x='County', y='Death_Rates', data=CDR, ax=axs[0, 3])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high Death Rates', ylabel='Crude Death Rates')
        # plt.title('Counties having high Crude Death Rates')

        # health facility delivery: percentage of babies delivered in a health facility delivery
        HFD = self.data.data2[['County', 'Healthcare_Facility_Density']].sort_values(
            'Healthcare_Facility_Density', ascending=False).head(7)
        ax = sns.barplot(x='County', y='Healthcare_Facility_Density', data=HFD, ax=axs[0, 4])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties having Low health facility delivery',
               ylabel='health facility delivery')

        # Fertility rate:  is the average number of children that would be
        # born to a woman over her lifetime
        FR = self.data.data2[['County', 'Fertility']].sort_values('Fertility',
                                                                  ascending=False).head(7)
        ax = sns.barplot(x='County', y='Fertility', data=FR, ax=axs[1, 0])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high fertility rates', ylabel='Fertility rate')

        # Average Household Size: Persons per household.
        AHS = self.data.data2[['County', 'Household_Size']].sort_values('Household_Size',
                                                                        ascending=False).head(7)
        ax = sns.barplot(x='County', y='Household_Size', data=AHS, ax=axs[1, 1])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high Average Household Size',
               ylabel='Average Household Size')
        # plt.title('Counties having high Average Household Size')

        # HIV Prevalence Rate: Percentage of people living with HIV.
        HIV = self.data.data2[['County', 'HIV_Prevalence']].sort_values('HIV_Prevalence',
                                                                        ascending=False).head(7)
        ax = sns.barplot(x='County', y='HIV_Prevalence', data=HIV, ax=axs[1, 2])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with High HIV Prevalence Rates', ylabel='HIV Prevalence Rate')

        # HIV Prevalence Rate: Percentage of people living with HIV.
        HIV = self.data.data2[['County', 'HIV_Prevalence']].sort_values('HIV_Prevalence',
                                                                        ascending=True).head(7)
        ax = sns.barplot(x='County', y='HIV_Prevalence', data=HIV, ax=axs[1, 3])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with Low HIV Prevalence Rates', ylabel='HIV Prevalence Rate')

        # Contraceptive prevalence: the proportion of women who are currently using,
        # or whose sexual partner is currently using at least one method of contraception,
        # regardless of the method being used.
        CP = self.data.data2[['County', 'Contraceptive_prevalence']].sort_values(
            'Contraceptive_prevalence', ascending=False).head(7)
        ax = sns.barplot(x='County', y='Contraceptive_prevalence', data=CP, ax=axs[1, 4])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high Contraceptive prevalence',
               ylabel='Contraceptive prevalence')

        # Unemployment rates: people of working age who are without work, are available for work,
        # and have taken specific steps to find work
        pop = self.data.data2[['County', 'Unemployment_Rate']].sort_values('Unemployment_Rate',
                                                                           ascending=False).head(7)
        ax = sns.barplot(x='County', y='Unemployment_Rate', data=pop, ax=axs[2, 0])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high Unemployment Rates', ylabel='Unemployment Rates')

        # Poverty Rate: ratio of the number of people (in a given age group) whose income
        # falls below the poverty line
        PR = self.data.data2[['County', 'Poverty_Rate']].sort_values('Poverty_Rate',
                                                                     ascending=False).head(7)
        ax = sns.barplot(x='County', y='Poverty_Rate', data=PR, ax=axs[2, 1])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high Poverty rates', ylabel='Poverty Rate')

        # Crime index:systematic, quantitative results about crime per 100,000 people
        CR = self.data.data2[['County', 'Crime_index']].sort_values('Crime_index',
                                                                    ascending=False).head(7)
        ax = sns.barplot(x='County', y='Crime_index', data=CR, ax=axs[2, 2])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high crime rates', ylabel='Crime index')

        # Crime index:systematic, quantitative results about crime per 100,000 people
        plt.figure(figsize=(30, 10))
        CR = self.data.data2[['County', 'Crime_index']].sort_values('Crime_index',
                                                                    ascending=True).head(7)
        ax = sns.barplot(x='County', y='Crime_index', data=CR, ax=axs[2, 3])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with low crime rates', ylabel='Crime index')

        # Child Marriage Prevalence Rate: any formal marriage or informal
        # union between a child under the age of 18
        # and an adult or another child
        CMPR = self.data.data2[['County', 'Child_Marriage_Prevalence']].sort_values(
            'Child_Marriage_Prevalence', ascending=False).head(7)
        ax = sns.barplot(x='County', y='Child_Marriage_Prevalence', data=CMPR, ax=axs[2, 4])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high Child Marriage Prevalence Rates',
               ylabel='Child Marriage Prevalence Rate')

        # Electricity access Rates: percentage of population with access to electricity.
        EAR = self.data.data2[['County', 'Electricity_access']].sort_values('Electricity_access',
                                                                            ascending=True).head(7)
        ax = sns.barplot(x='County', y='Electricity_access', data=EAR, ax=axs[3, 0])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with low lectricity access Rates',
               ylabel='Electricity access Rates')

        # Electricity access Rates: percentage of population with access to electricity.
        EAR = self.data.data2[['County', 'Electricity_access']].sort_values('Electricity_access',
                                                                            ascending=False).head(7)
        ax = sns.barplot(x='County', y='Electricity_access', data=EAR, ax=axs[3, 1])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with High lectricity access Rates',
               ylabel='Electricity access Rates')

        # Urbanization Rate: Percentage of the total population living in urban areas
        UB = self.data.data2[['County', 'Urbanization']].sort_values('Urbanization',
                                                                     ascending=True).head(7)
        ax = sns.barplot(x='County', y='Urbanization', data=UB, ax=axs[3, 2])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with low Urbanization Rates', ylabel='Urbanization Rate')

        # Urbanization Rate: Percentage of the total population living in urban areas
        UB = self.data.data2[['County', 'Urbanization']].sort_values('Urbanization',
                                                                     ascending=False).head(7)
        ax = sns.barplot(x='County', y='Urbanization', data=UB, ax=axs[3, 3])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with High Urbanization Rates', ylabel='Urbanization Rate')

        # Education level:  Percent Reached Secondary school or higher
        EL = self.data.data2[['County', 'Education_level']].sort_values('Education_level',
                                                                        ascending=True).head(7)
        ax = sns.barplot(x='County', y='Education_level', data=EL, ax=axs[3, 4])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with low education levels', ylabel='Education level')

        # Population: Number of people in the county
        P = self.data.data2[['County', 'Population_size']].sort_values('Population_size',
                                                                       ascending=False).head(7)
        ax = sns.barplot(x='County', y='Population_size', data=P, ax=axs[4, 0])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high population', ylabel='Population ')

        # Population Growth Rates: increase in the number of people in
        # a population or dispersed group.
        plt.figure(figsize=(30, 10))
        PGR = self.data.data2[['County', 'Growth_Rates']].sort_values('Growth_Rates',
                                                                      ascending=False).head(7)
        ax = sns.barplot(x='County', y='Growth_Rates', data=PGR, ax=axs[4, 1])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
        ax.set(xlabel='Counties with high population growth rates',
               ylabel='Population Growth Rates')

        # Area (Square km): The size of the county
        Area = self.data.data2[['County', 'Land_size']].sort_values('Land_size',
                                                                    ascending=False).head(7)
        ax = sns.barplot(x='County', y='Land_size', data=Area, ax=axs[4, 2])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
            ax.set(xlabel='Counties with large area size', ylabel='Area (Square km)')

        # Average GDP: county contribution to National GDP
        AGDP = self.data.data2[['County', 'Gross_County_Product']].sort_values(
            'Gross_County_Product', ascending=False).head(7)
        ax = sns.barplot(x='County', y='Gross_County_Product', data=AGDP, ax=axs[4, 3])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
            ax.set(xlabel='Counties with high GDP', ylabel='GDP')

        # Average GDP: percentage county contribution to National GDP
        AGDP = self.data.data2[['County', 'Gross_County_Product']].sort_values(
            'Gross_County_Product', ascending=True).head(7)
        ax = sns.barplot(x='County', y='Gross_County_Product', data=AGDP, ax=axs[4, 4])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
            ax.set(xlabel='Counties with low GDP', ylabel='GDP')

        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)

        plt.tight_layout()
        plt.savefig('figs/EDA.pdf')









