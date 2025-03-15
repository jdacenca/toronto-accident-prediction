import numpy as np
import pandas as pd
from DataProcessor import data_overview,data_cleaning,sample_and_update_data, data_preprocessing_svm, train_and_evaluate_model, grid_search_svm
from Visualizer import scatter_plot, pie_chart, bar_chart, heatmap_correlation
from Visualizer import  heatmap_missing_values, hist_plot, pair_plot, spline_plot, cat_distribution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Random seed for reproducibility
np.random.seed(17)

try:
    data_ksi = pd.read_csv('./data/Total_KSI.csv')
    
    data_overview(data_ksi)
   
    # Drop unnecessary columns
    data_ksi.drop(columns=['INDEX','OBJECTID','HOOD_158','HOOD_140','OFFSET', 'STREET1', 'STREET2','NEIGHBOURHOOD_158','NEIGHBOURHOOD_140'], inplace=True)

    # Columns to drop
    columns_to_drop = ['FATAL_NO', 'DISTRICT', 'DIVISION','x','y','INJURY','INVTYPE','INVAGE','INITDIR','VEHTYPE',
                           'MANOEUVER','DRIVACT','PEDTYPE','PEDACT','CYCLISTYPE','CYCACT','CYCCOND','PEDCOND','DRIVCOND']
    
    # Visualizations
    # scatter_plot(data_ksi,"./images/scatter_plot.png")
    # pie_chart(data_ksi,"./images/pie_chart.png")
    # bar_chart(data_ksi,"./images/bar_chart.png")
    # heatmap_correlation(data_ksi,"./images/heatmap_correlation.png")
    # heatmap_missing_values(data_ksi,"./images/heatmap_missing_values.png")
    # hist_plot(data_ksi,"./images/hist_plot.png")
    # pair_plot(data_ksi,"./images/pair_plot.png")
    # cat_distribution(data_ksi,"./images/cat_distribution.png")
    # spline_plot(data_ksi,"./images/spline_plot.png")
    
    cleaned_df = data_cleaning(data_ksi,columns_to_drop,class_imb='oversampling')
    # data_overview(cleaned_df)

    pie_chart(cleaned_df,"./images/pie_chart.png")
  
    # Split the data into features and target
    unseen_features, unseen_labels, cleaned_df, features, target = sample_and_update_data(cleaned_df)
        
    # Encode the target variable 
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)

    # Split the data into train & test
    X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.2, random_state=17)

    # Preprocess the data
    pipeline_svm = data_preprocessing_svm(features, smote=False)

    # list of dictionaries containing the parameters to be tested
    param_grid_svm_linear = [
        {'svm__kernel': ['linear'], 'svm__C': [1, 10, 100]}]

    param_grid_svm_rbf = [        
        {'svm__kernel': ['rbf'], 'svm__C': [0.1, 1, 10, 100], 'svm__gamma': [0.03, 0.3, 1.0, 3.0]} # rbf kernel rank#1 C=1, gamma=3.0
    ]

    param_grid_svm_poly= [
        {'svm__kernel': ['poly'], 'svm__C': [0.1, 1, 10], 'svm__gamma': [0.1, 0.3, 1.0, 3.0], 'svm__degree': [2, 3]}  # poly kernel 
    ]
    

    # Train and evaluate the model
    grid_search = grid_search_svm(pipeline_svm, param_grid_svm_linear)
    train_and_evaluate_model("svm_linear",grid_search, X_train, y_train, X_test, y_test, unseen_features, unseen_labels)
    
    # Train and evaluate the model
    grid_search = grid_search_svm(pipeline_svm, param_grid_svm_rbf)
    train_and_evaluate_model("svm_rbf",grid_search, X_train, y_train, X_test, y_test, unseen_features, unseen_labels)
    
    # Train and evaluate the model
    grid_search = grid_search_svm(pipeline_svm, param_grid_svm_poly)
    train_and_evaluate_model("svm_poly",grid_search, X_train, y_train, X_test, y_test, unseen_features, unseen_labels)
    

except Exception as e:
    print("\n===================== ERROR =====================")
    print(f"An error occurred : {e}")
 
