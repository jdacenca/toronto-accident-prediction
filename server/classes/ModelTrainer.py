from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class ModelTrainer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = None

    def train_model(self):
        num_features = self.data_processor.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_features = self.data_processor.X.select_dtypes(include=['object']).columns.tolist()

        num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown="ignore"))])

        preprocessor = ColumnTransformer([('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])

        pipe_svm = Pipeline([('preprocessor', preprocessor), ('svm', SVC(random_state=17))])

        param_grid = [{'svm__kernel': ['poly'], 'svm__C': [0.1], 'svm__gamma': [3.0], 'svm__degree': [3]}]
        grid_search = GridSearchCV(estimator=pipe_svm, param_grid=param_grid, scoring='accuracy', refit=True, verbose=3)
        grid_search.fit(self.data_processor.X_train, self.data_processor.y_train)

        self.model = grid_search.best_estimator_

    def predict(self):
        return self.model.predict(self.data_processor.X_test)
