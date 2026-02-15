# Task 1: Data Cleaning & Preprocessing

## Project Objective
The goal of this task is to learn how to prepare raw, messy data for Machine Learning models. I utilized the Titanic Dataset to practice handling missing information, encoding categories, and feature scaling.

## Workflow
* **Exploration**: Checked for null values and data types to understand the dataset structure.
* **Handling Nulls**: Imputed missing numerical values using the median and categorical values using the mode.
* **Categorical Encoding**: Converted features like 'Sex' into numerical values using Label Encoding to make them model-ready.
* **Feature Scaling**: Applied Standardization to numerical features like 'Age' and 'Fare' to ensure they are on a similar scale.
* **Outlier Removal**: Used boxplots to identify extreme values and filtered them using the Interquartile Range (IQR) method.

## Tools Used
* Python
* Pandas & NumPy
* Scikit-Learn (StandardScaler, LabelEncoder)