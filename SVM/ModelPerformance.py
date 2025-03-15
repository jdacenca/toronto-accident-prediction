import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

class ModelPerformance:

    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def conf_matrix(self, save_path):
        cm = confusion_matrix(self.y_test, self.model.predict(self.x_test))
        cm_df = pd.DataFrame(cm, index=np.unique(self.y_test), columns=np.unique(self.y_test))
        plt.figure(figsize=(6, 5))
        custom_cmap = sns.light_palette("#F6866A", as_cmap=True)
        sns.heatmap(cm_df, annot=True, fmt="d", cmap=custom_cmap)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(save_path)
        plt.show()
