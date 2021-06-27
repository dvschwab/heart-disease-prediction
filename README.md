# heart-disease-prediction

A logistic regression model to predict heart disease from 13 features

This model predicts the presence or absence of heart disease based on 13 medical and demographic features. It uses data from https://www.kaggle.com/ronitf/heart-disease-uci, which was published by researchers in de-identified form. There are 296 cases total, of which 136 indicate heart disease.

The notebook was built with Python 3.8.10. The YAML file lists the required packages and may be used to create a conda environment with these packages: see `RUNNING_THE_MODEL` for details.

For questions or comments, please contact me at dvschwab@protonmail.com.

## Summary

The notebook fits 3 versions of a logistic regression model:
> * a full model with all cases and features
> * a reduced model containing only the 9 features with the greatest predictive value
> * a reduced model with 20 outliers trimmed.

All perform well, with a median accuracy of 83% for the test data. None of the models are prone to false positives, although each is slightly better at predicting the absence of heart disease than its presence. However, the cross-validation scores show sample dependence for each model, as shown below for the reduced model:

>> Cross-Validation (5 splits)  
>> Test Accuracy: [0.75, 0.83, 0.90, 0.90, 0.86]  
>> Test MSE:      [0.25, 0.17, 0.10, 0.10, 0.14]

All results are for the test data; *accuracy* is the mean accuracy for both targets, while *MSE* is the mean-squared error. Each split contained 59 or 60 cases. The sample dependence is clear from the first, third, and fourth splits.

The relatively small number of cases per model feature is another limitation. For the reduced model containing 9 features and using a 30/70 test-train split, this amounts to 89 cases / 9 features = 10 cases per feature (standard rounding). Most researchers consider this the minimum ratio of cases-to-feature, as the coefficient standard errors will likely be too large to evaluate the relative importance of each feature.

## Feature Descriptions

For clarity, only the reduced model with 9 predictors is described from this point on. Full descriptions of all three models are available in the main notebook `heart-disease-prediction-data-analysis.ipynb`.

These are the demographic and numeric (i.e. continuous) features used in the model:

> * age: age of subject
> * sex: sex of subject (Male/Female)
> * max_heart_rate: maximum heart rate
> * oldpeak: ST depression from exercise

These are the categorical features. Each one indicates the presence of the named indicator, except for *flouroscopy*, which indicates how many major vessels were colored by a specific flouroscopic test (Zero or Nonzero). In all cases, Yes indicates an abnormal condition while No indicates normal functioning.

> * chest_pain: chest pain (Yes/No)
> * angina: exercise-induced angina (Yes/No)
> * ST_slope: abnormal ST slope (Yes/No)
> * flouroscopy: number of vessels colored by flouroscopy (Zero/Nonzero)
> * heart_defect: heart defect (Yes/No)

The target variable, *heart_disease*, is coded *Yes* or *No*. The data for the model presented here has 160 cases without heart disease and 136 with heart disease, totalling 296 cases in all.

## Representative Data Sample

This is a representative sample of 10 cases from the data frame used to fit the reduced model. The categorical features have been replaced with text to make them more understandable.

![](Images/heart_df_present.png)

## Descriptive Statistics

These are three descriptive statistics estimated for the model. The first figure presents summary statistics for the four numeric features. The remaining figures present a heatmap of the cross-correlations between these features and a representative boxplot comparing the presence of heart disease vs. the maximum heart rate.

### Summary of Numeric Features


Summary statistics for the numeric variables are unexceptional. This may be due to the fact that the features are well-established medical measurement that are known to be at least approximately normal for large samples.

![](Images/heart_disease_summary.png)

### Correlation of Numeric Features

The correlation between the three numeric features in the reduced model is shown here as a heatmap. As with a traditional correlation matrix, the diagonal elements show each feature's correlation with itself and are therefore unimportant. The remaining elements show the cross-correlation between the row feature and the column feature: for example, the element at the upper right is the correlation between the features *age* and *oldpeak*

The scale on the right-hand side ranges from perfect positive correlation (at the top) to perfect negative correlation (at the bottom). As can be seen, blue elements show positive correlation and green elements show negative correlation; in both cases, darker colors indicate a stronger correlation. Here, we se a small positive correlation between the features *age* and *old_peak* (approx. 0.25) and a small negative correlation between the remaining two features (approx. -.025). Since these features are included in the model, the presence of small correlations is desirable so that the effect of each feature on the target is relatively independent.

![](Images/heart_corr_heatmap.png)

### Boxplot of Maximum Heart Rate vs. Heart Disease

This boxplot shows the relationship between the feature *max_heart_rate* and the target. It is representative of the relationships between each numeric feature and the target. Note that the 20 outliers have been excluded from the plot to aid in readability.

As can be seen, patients with heart disease have a relatively lower maximum heart rate: the median value is approximately 20 beats less for patients with heart disease. This indicates the feature may be a good predictor of heart disease.

![](Images/boxplot_heart_rate_vs_disease.png)

## Model Estimation and Results