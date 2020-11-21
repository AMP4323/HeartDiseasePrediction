#!/usr/bin/env python
# coding: utf-8

# Author : Jatin Luthra (18UCC082)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
import joblib


# Initializations

df = pd.read_csv('Datasets/heart_new.csv')
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

sns.set_style('whitegrid')

# Data Preprocessing

X = df.copy()
ss = StandardScaler()
ss.fit(X[num_cols])

X[num_cols] = ss.transform(X[num_cols])


# Print Columns, Frequencies and DataTypes

print(df.info())


# Statistic Descriptions of Numerical attributes

print(df[num_cols].describe())


# Boxplot of all the Numeric Columns

f, ax = plt.subplots(figsize=(15, 10))
ax = sns.boxplot(data=df[num_cols], palette='bright')
ax.set_xlabel("Numeric Columns")
ax.set_ylabel("Value")
ax.set_title("BoxPlot for Numeric Columns")
plt.ylim(0, None)
plt.savefig('Figures/boxplot.svg')
plt.show()


# Boxplot after Standardization of Numerical attributes

f, ax = plt.subplots(figsize=(15, 10))
ax = sns.boxplot(data=X[num_cols], palette='bright')
ax.set_xlabel("Numeric Columns")
ax.set_ylabel("Standardized Value")
ax.set_title("Standard BoxPlot for Numeric Columns")
plt.savefig('Figures/boxplot_standard.svg')
plt.show()


# Plot the Correlation Heatmap among all attributes

corr = df.corr()

f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr, annot=True, fmt='.2f')
ax.set_xlabel("Attributes")
ax.set_ylabel("Attributes")
ax.set_title("Correlation Heatmap")
plt.savefig('Figures/heatmap.svg')
plt.show()


# Plot all attributes paired with the target variable, colourfully distinctive by sex variable

sns.pairplot(df, y_vars=['target'], hue='sex', size = 5)


# Plot the Age distribution and the estimated PDF

f, ax = plt.subplots(figsize=(15, 10))
ax = sns.histplot(df.age, color='red', kde=True)
ax.set_title("Age Distribution")
plt.savefig('Figures/hist.svg')
plt.show()


# # Plot the Standardized Age distribution and the estimated PDF

f, ax = plt.subplots(figsize=(15, 10))
ax = sns.histplot(X.age, color='red', kde=True,)
ax.set_title("Age Distribution after Standardization")
plt.savefig('Figures/hist_std.svg')
plt.show()


# Aggregate the Age variable into Age Ranges, Plot a Violinplot against the Target variable, colourfully distinctive by sex variable

f, ax = plt.subplots(figsize=(15, 10))

bins = list(range(25,85,10))
bin_labels = ["25-34", "35-44", "45-54", "55-64", "65-74"]

ax = sns.violinplot(data=df, x=pd.cut(df['age'], bins, labels=bin_labels, include_lowest=True), y='target', hue='sex', palette=["#FF0D57", "#1E88E5"])

handles, labels = plt.gca().get_legend_handles_labels()
labels = ['Female', 'Male']
plt.legend(handles,labels)

ax.set_xlabel("Age Range (Bin_Size=10)")
ax.set_ylabel("Target")
ax.set_title("Age vs Heart Disease segregated by Sex")
plt.savefig('Figures/violin.svg')
plt.show()


# Plot the PieChart of Sex Distribution

f, ax = plt.subplots(figsize=(15, 10))
ax.pie([sum(df['sex'] == 0), sum(df['sex'] == 1)], labels=['Female', 'Male'], autopct='%1.2f%%', colors=["#FF0D57", "#1E88E5"])
ax.set_title("Pie Chart for Sex Distribution")
plt.savefig('Figures/pie.svg')
plt.show()
