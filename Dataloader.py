import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    The DataLoader object manages getting all data necessary for simulation

    Member variables:
        *._data_file     data file paths
        *._data          Loaded data
    """

    def __init__(self, data_path="/Users/user/PycharmProjects/kenya economic indicators/data/data.xls",
                       geo_data_path="/Users/user/PycharmProjects/kenya economic indicators/data/geodata.xls"):

        self.data = None
        self.data2 = None
        self.county_data_scaled = []
        self.county_names = []
        self.geo_data = None
        self.data_path = data_path
        self.geo_data_path = geo_data_path
        self.get_data()
        self.scale_data()
        self._get_geo_data()

    def get_data(self):
        """
        Main function for social economic data loading
        :return: data
        """
        # use pandas to load the social_economic data

        data = pd.read_excel(self.data_path, index_col=0)
        data2 = pd.read_excel(self.data_path)

        self.county_names = data2["County"]
        self.data = data
        self.data2 = data2
        return self.data, self.county_names

    def _get_geo_data(self):
        # load geodata
        self.geo_data = pd.read_excel(self.geo_data_path)
        return self.geo_data

    def scale_data(self):
        """
        Scales the socioeconomic data
        :return: ndarray of the scaled data
        """
        scaler = StandardScaler()
        county_data_scaled = scaler.fit_transform(self.data)
        self.county_data_scaled = county_data_scaled
        return self.county_data_scaled
