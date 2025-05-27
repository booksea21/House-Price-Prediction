**Predicting Real Estate Prices Using Feature Engineering and Linear Regression**

This project uses the Ames Housing dataset to build a machine learning model that predicts housing prices in Ames, Iowa. The dataset contains detailed records on 80 variables describing aspects of residential homes.

The goal is to apply solid data cleaning, preprocessing, and modeling practices to generate accurate predictions of house sale prices.

**Key Features**
-Extensive Data Cleaning: Missing values are handled based on data context (e.g., 'None' for categorical features, median/mode for numerical).

-Feature Engineering: Creation of new features like total square footage, total bathrooms, and transformation of skewed data using Box-Cox.

-Categorical Encoding: Uses both ordinal and label encoding to convert textual data into usable numerical values.

-Skewness Adjustment: Box-Cox transformations applied to reduce feature skewness for better linear modeling.

-Linear Regression Model: Simple yet powerful linear model trained and evaluated using RMSE and explained variance.

-Final Prediction Output: Model is used to generate submission-ready predictions for the test dataset.

