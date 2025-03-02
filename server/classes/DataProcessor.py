import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path):
        self.data_ksi = pd.read_csv(data_path)
        self.clean_data()

    def clean_data(self):
        # Drop unnecessary columns
        self.data_ksi = self.data_ksi.drop(columns=['INDEX', 'ACCNUM', 'OBJECTID', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'STREET1', 'STREET2', 'OFFSET', 'FATAL_NO', 'DISTRICT', 'DIVISION'])

        # Separate the features & target
        self.y = self.data_ksi["ACCLASS"]
        self.X = self.data_ksi.drop(columns=["ACCLASS"])

        # Encode target variable
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=17)
