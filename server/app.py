from flask import Flask, jsonify
from flask_cors import CORS

from sklearn.metrics import confusion_matrix
from classes.DataProcessor import DataProcessor

from classes.Visualizer import Visualizer
from classes.ModelTrainer import ModelTrainer
from classes.ModelPerformance import ModelPerformance

app = Flask(__name__)
CORS(app)  # Allow React to call Flask API

data_processor = DataProcessor('./data/Total_KSI.csv')
visualizer = Visualizer(data_processor.data_ksi)
model_trainer = ModelTrainer(data_processor)

# Visualizations
visualizer.scatter_plot('../client/public/images/scatterplot.png')
visualizer.pie_chart('../client/public/images/piechart.png')
visualizer.bar_chart('../client/public/images/barchart.png')

# Train the model
model = model_trainer.train_model()
performance = ModelPerformance(model,data_processor.X_test, data_processor.y_test)

performance.conf_matrix('../client/public/images/confusion_matrix_svm.png')
performance.plot_classification_report_radial('../client/public/images/classification_report_svm.png')

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({
        "conf_matrix": './images/confusion_matrix_svm.png',
        "scatter_plot": './images/scatterplot.png',
        "pie_chart": './images/piechart.png'
    })

if __name__ == '__main__':
    app.run(debug=True)