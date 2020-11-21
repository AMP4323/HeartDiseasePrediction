#!/usr/bin/env python
# coding: utf-8

# Author : Jatin Luthra (18UCC082)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import shap


# All the scikit-learn imports

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2


# Initializations


pd.set_option('display.max_columns', 50)
df = pd.read_csv('Datasets/heart_new.csv')                                  # Read the Dataset

model_results = []
saves_dir = "Saves/"

cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


# Data Preprocessing

y = df['target']
X = pd.get_dummies(df.drop('target', axis=1), columns=cat_cols)             # Dummy Encoding Categorical Variables


ss = StandardScaler()                                                       # Standardization
ss.fit(X[num_cols])
X[num_cols] = ss.transform(X[num_cols])


# Copy dataset and add details to the values

df_explained = df.copy()
df_explained.loc[df['sex']==0, 'sex'] = 'female'
df_explained.loc[df['sex']==1, 'sex'] = 'male'

df_explained.loc[df['cp']==0, 'cp'] = 'typical_angina'
df_explained.loc[df['cp']==1, 'cp'] = 'atypical_angina'
df_explained.loc[df['cp']==2, 'cp'] = 'non_anginal_pain'
df_explained.loc[df['cp']==3, 'cp'] = 'asymptomatic'

df_explained.loc[df['restecg']==0, 'restecg'] = 'normal'
df_explained.loc[df['restecg']==1, 'restecg'] = 'st_t_wave_abnormality'
df_explained.loc[df['restecg']==2, 'restecg'] = 'left_ventricular_hypertrophy'

df_explained.loc[df['slope']==0, 'slope'] = 'upsloping'
df_explained.loc[df['slope']==1, 'slope'] = 'flat'
df_explained.loc[df['slope']==2, 'slope'] = 'downsloping'

df_explained.loc[df['thal']==0, 'thal'] = 'na'
df_explained.loc[df['thal']==1, 'thal'] = 'fixed_defect'
df_explained.loc[df['thal']==2, 'thal'] = 'normal'
df_explained.loc[df['thal']==3, 'thal'] = 'reversible_defect'

df_explained = pd.get_dummies(df_explained, columns=cat_cols)


# Split the Dataset into Train and Test (28.5%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.285, shuffle=True, random_state=42)


# Naive Bayes

nb = GaussianNB()
nb.fit(X_train, y_train)

model_results.append(['Naive Bayes', nb.score(X_train, y_train), nb.score(X_test, y_test), ""])

joblib.dump(nb, saves_dir + 'naive_bayes.model')


# Logistic Regression:

lr = LogisticRegression()
params_grid = {
    'C' : np.logspace(-4, 4)
}
cv = GridSearchCV(lr, params_grid, n_jobs=-1, cv=5, verbose=1)
cv.fit(X_train, y_train)

model_results.append(['Logistic Regression', cv.best_score_, cv.best_estimator_.score(X_test, y_test), "C = " + str(round(cv.best_params_['C'],2))])

joblib.dump(cv.best_estimator_, saves_dir + 'logistic_regression.model')


# Plot the GridSearchCV Results

plt.semilogx(np.logspace(-4, 4), cv.cv_results_['mean_test_score'])
plt.title("Logistic Regression Comparison")
plt.xlabel("C")
plt.ylabel("CV=5 Score")
plt.savefig('Figures/log_comp.svg')
plt.show()


# Ridge Regression

rc = RidgeClassifier()

params_grid = {
    'alpha': np.logspace(-9, 3)
}
cv = GridSearchCV(rc, params_grid, n_jobs=-1, cv=5, verbose=1)
cv.fit(X_train, y_train)

model_results.append(['Ridge Regression', cv.best_score_, cv.best_estimator_.score(X_test, y_test), "alpha = " + str(round(cv.best_params_['alpha'], 5))])

joblib.dump(cv.best_estimator_, saves_dir + 'ridge_regression.model')


# Plot the GridSearchCV Results

plt.semilogx(np.logspace(-9, 3), cv.cv_results_['mean_test_score'])
plt.title("Ridge Regression Comparison")
plt.xlabel("Alpha")
plt.ylabel("CV=5 Score")
plt.savefig('Figures/ridge_comp.svg')
plt.show()


# K-Nearest Neighbors

knn = KNeighborsClassifier()
params_grid = {
    'n_neighbors': range(1,30)
}
cv = GridSearchCV(knn, params_grid, n_jobs=-1, cv=5, verbose=1)
cv.fit(X_train, y_train)

model_results.append(['K-Neighbors', cv.best_score_, cv.best_estimator_.score(X_test, y_test), "n_neighbors = " + str(cv.best_params_['n_neighbors'])])

joblib.dump(cv.best_estimator_, saves_dir + 'k_neighbors.model')


# Plot the GridSearchCV Results

plt.plot(range(1,30), cv.cv_results_['mean_test_score'])
plt.title("K-Neighbors Comparison")
plt.xlabel("Neighbors (N)")
plt.ylabel("CV=5 Score")
plt.savefig('Figures/knn_comp.svg')
plt.show()


# Support Vector Machines

svc = SVC()
params_grid = {
    'C' : np.logspace(-4, 4, 9),
    'gamma' : np.logspace(-4, 4, 9),
    'kernel' : ['poly'],
    'degree' : range(2,7)
}
cv = GridSearchCV(svc, params_grid, n_jobs=-1, cv=5, verbose=1)
cv.fit(X_train, y_train)

model_results.append(['SVM', cv.best_score_, cv.best_estimator_.score(X_test, y_test), "C = " + str(cv.best_params_['C'])])

joblib.dump(cv.best_estimator_, saves_dir + 'svm.model')


# SVM with Stochastic Gradient Descent

sgd = SGDClassifier(random_state=43)

params_grid = {
    'alpha': np.logspace(-3, 0, 200),
}
cv = GridSearchCV(sgd, params_grid, n_jobs=-1, cv=5, verbose=1)
cv.fit(X_train, y_train)

model_results.append(['Stochastic Gradient Descent', cv.best_score_, cv.best_estimator_.score(X_test, y_test), "alpha = " + str(round(cv.best_params_['alpha'], 5))])

joblib.dump(cv.best_estimator_, saves_dir + 'sgd.model')


# Plot the GridSearchCV Results

plt.semilogx(np.logspace(-3, 0, 200), cv.cv_results_['mean_test_score'])
plt.title("Stochastic Gradient Descent Comparison")
plt.xlabel("Alpha")
plt.ylabel("CV=5 Score")
plt.savefig('Figures/sgd_comp.svg')
plt.show()


# Decision Trees

dtc = DecisionTreeClassifier(random_state=42)
params_grid = {
      "criterion" : ["gini", "entropy"], 
      "splitter" : ["best", "random"], 
      "max_depth" : range(1,20), 
      "min_samples_split" : range(2, 5), 
      "min_samples_leaf" : range(1, 20)
}

cv = GridSearchCV(dtc, params_grid, n_jobs=-1, cv=5, verbose=1)
cv.fit(X_train, y_train)

model_results.append(['Decision Trees', cv.best_score_, cv.best_estimator_.score(X_test, y_test), "max_depth = " + str(cv.best_params_['max_depth'])])

joblib.dump(cv.best_estimator_, saves_dir + 'decision_trees.model')


# Plot the Decision Tree

plt.figure(figsize=(25,20))
plot_tree(cv.best_estimator_, feature_names=X.columns,  class_names=['0', '1'], filled=True)
plt.savefig('Figures/dct.svg')
plt.show()


# Random Forests

rf = RandomForestClassifier(random_state=42)
params_grid = {
    'n_estimators': range(10,300,10),
}

cv = GridSearchCV(rf, params_grid, n_jobs=-1, cv=5, verbose=1)
cv.fit(X_train, y_train)

model_results.append(['Random Forests', cv.best_score_, cv.best_estimator_.score(X_test, y_test), "n_estimators = " + str(cv.best_params_['n_estimators'])])

joblib.dump(cv.best_estimator_, saves_dir + 'random_forests.model')


# Plot the GridSearchCV Results

plt.plot(range(10,300,10), cv.cv_results_['mean_test_score'])
plt.title("Random Forests Comparison")
plt.xlabel("Estimators")
plt.ylabel("CV=5 Score")
plt.savefig('Figures/rf_comp.svg')
plt.show()


# Aggregate all the results

results = pd.DataFrame(model_results, columns=['Algorithm', 'Train Accuracy (%)', 'Test Accuracy (%)', 'Best Parameters'])

results.loc[:][['Train Accuracy (%)', 'Test Accuracy (%)']] = results.loc[:][['Train Accuracy (%)', 'Test Accuracy (%)']]  * 100

results = results.round({'Train Accuracy (%)' : 2, 'Test Accuracy (%)' : 2})


# Save the results as CSV file:

results.to_csv('results.csv', index=False)


# Export the parameters for Data Preprocessing in JavaScript

export_params = {
    'cat_cols' : cat_cols,
    'num_cols' : num_cols,
    'ss_mean' : ss.mean_.tolist(),
    'ss_std' : np.sqrt(ss.var_).tolist(),
    'cols' : X.columns.tolist()
}

with open("params.json", "w") as fp:
    fp.write(json.dumps(export_params))


# Plot the Feature Importance of Random Forests:

explainer = shap.TreeExplainer(cv.best_estimator_)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, feature_names = df_explained.columns, show=False)
plt.title("Feature Importance by SHAP Values")
plt.savefig('Figures/feature_importance.svg')


# Feature Selection on All attributes:

best_features = []
best_features_comp = []
for i in range(3,14):
    y = df['target']
    X = df.drop('target', axis=1)
    feat_select = SelectKBest(chi2, k=i)
    feat_select.fit(X, y)
    X_new = pd.DataFrame(feat_select.transform(X), columns=X.columns[feat_select.get_support()])

    num_cols_new = [col for col in X_new.columns if col in num_cols]
    cat_cols_new = [col for col in X_new.columns if col in cat_cols]

    ss = StandardScaler()
    ss.fit(X_new[num_cols_new])


    X_new[num_cols_new] = ss.transform(X_new[num_cols_new])
    
    X = pd.get_dummies(X_new, columns=cat_cols_new)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.285, shuffle=True, random_state=42)
    
    knn = KNeighborsClassifier()
    params_grid = {
        'n_neighbors': range(1,30)
    }
    
    cv = GridSearchCV(knn, params_grid, n_jobs=-1, cv=5, verbose=0)
    cv.fit(X_train, y_train)
    best_features.append([cv.best_score_,cv.best_estimator_.score(X_test, y_test)])
    
best_idx = np.argmax(np.array(best_features)) // 2
best_features_comp.append(["KNN All"] + (np.round(np.array(best_features[best_idx])*100,2)).tolist() + [best_idx + 3])


# Plot the Results

plt.plot(range(3,14), best_features)
plt.title("Best Features for KNN (All)")
plt.xlabel("No. of Features")
plt.ylabel("CV=5 Score")
plt.legend(["Train Acc.", "Test Acc."])
plt.savefig('Figures/best_feat_all.svg')
plt.show()


# Feature Selection on Categorical attributes:

best_features = []
for i in range(1,9):
    y = df['target']
    X = df[cat_cols]
    feat_select = SelectKBest(chi2, k=i)
    feat_select.fit(X, y)
    X_new = pd.DataFrame(feat_select.transform(X), columns=X.columns[feat_select.get_support()])

    cat_cols_new = [col for col in X_new.columns if col in cat_cols]

    X = pd.get_dummies(X_new, columns=cat_cols_new)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.285, shuffle=True, random_state=42)
    
    knn = KNeighborsClassifier()
    params_grid = {
        'n_neighbors': range(1,30)
    }
    
    cv = GridSearchCV(knn, params_grid, n_jobs=-1, cv=5, verbose=0)
    cv.fit(X_train, y_train)
    best_features.append([cv.best_score_,cv.best_estimator_.score(X_test, y_test)])
    
best_idx = np.argmax(np.array(best_features)) // 2
best_features_comp.append(["KNN Categorical Only"] + (np.round(np.array(best_features[best_idx])*100,2)).tolist() + [best_idx + 1])


# Plot the Results

plt.plot(range(1,9), best_features)
plt.title("Best Features for KNN (Categorical Only)")
plt.xlabel("No. of Features")
plt.ylabel("CV=5 Score")
plt.legend(["Train Acc.", "Test Acc."])
plt.savefig('Figures/best_feat_cat.svg')
plt.show()


# Feature Selection on Numerical attributes:

best_features = []
for i in range(1,6):
    y = df['target']
    X = df[num_cols]
    feat_select = SelectKBest(chi2, k=i)
    feat_select.fit(X, y)
    X_new = pd.DataFrame(feat_select.transform(X), columns=X.columns[feat_select.get_support()])

    num_cols_new = [col for col in X_new.columns if col in num_cols]

    ss = StandardScaler()
    ss.fit(X_new[num_cols_new])


    X_new[num_cols_new] = ss.transform(X_new[num_cols_new])
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.285, shuffle=True, random_state=42)
    
    knn = KNeighborsClassifier()
    params_grid = {
        'n_neighbors': range(1,30)
    }
    
    cv = GridSearchCV(knn, params_grid, n_jobs=-1, cv=5, verbose=0)
    cv.fit(X_train, y_train)
    best_features.append([cv.best_score_,cv.best_estimator_.score(X_test, y_test)])
    
best_idx = np.argmax(np.array(best_features)) // 2
best_features_comp.append(["KNN Numerical Only"] + (np.round(np.array(best_features[best_idx])*100,2)).tolist() + [best_idx + 1])


# Plot the Results

plt.plot(range(1,6), best_features)
plt.title("Best Features for KNN (Numerical Only)")
plt.xlabel("No. of Features")
plt.ylabel("CV=5 Score")
plt.legend(["Train Acc.", "Test Acc."])
plt.savefig('Figures/best_feat_num.svg')
plt.show()


# Aggregate and Save the results to a CSV File

best_feat_results = pd.DataFrame(best_features_comp, columns=['Features Type', 'Training Accuracy (%)', 'Testing Accuracy (%)', 'No. of Features Kept'])
best_feat_results.to_csv('best_feat_results.csv', index=False)
