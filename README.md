# HeartDiseasePrediction
This project aims to predict in real-time if a patient has heart disease based on the test results and values.

## Website
The root of this repository contains the source of the website. It has a basic and clean UI to enter the values of test results and get the prediction. There are also few points of guidance based on the numerical values in the results. A hashing mechanism is also implemented so that the same result can't be shown after 5 seconds to prevent discrepancies and false reports.
The heart of these predictions is a K-Nearest Neighbour model trained on the UCI Heart Disease Prediction Dataset which is ported to JS using `sklearn-porter` library.

## Model Codes

All the codes used to train model and visualize the data are saved in the **src** folder.

### Libraries Used

Following libraries are used in the project:
    
* numpy
* pandas
* scikit-learn
* joblib
* sklearn-porter
* jupyter-notebooks

These libraries can easily be installed with the *pip* tool as follows:
```
pip install numpy pandas scikit-learn joblib sklearn-porter notebook
```

### Folder Structure

* Datasets

    This folder contains all the datasets used or referenced during the project. They can be found in the following links:
    
    * [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
    * [Kaggle Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)
    * [Kaggle Heart Disease Dataset](https://www.kaggle.com/johnsmith88/heart-disease-dataset)

* Figures

    This folder contains all the plots and figures in SVG format for ease of reference

* Saves

    This folder contains all the pickled trained models exported by *joblib* so that each model can be used by just importing them instead of training them again

* model_train

    model_train.ipynb is the Jupyter notebook which was used to write and experiment with the model training code. All the outputs are saved and can be rendered without running it by just opening it on Github.

    model_train.py is the simplified and commented version of the above notebook as a complete python script.

* EDA

    EDA.ipynb is the Jupyter notebook which was used to write and visualize the data and related plots of Data Insights. All the outputs are saved and can be rendered without running it by just opening it on Github.

    EDA.py is the simplified and commented version of the above notebook as a complete python script.

* outputs

    results.csv and best_feat_results.csv contain the final comparison of accuracies of all the models without and with feature selection respectively.

    params.json contains a dictionary of values necessary for data preprocessing in JS.