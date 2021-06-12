# heart-disease-prediction
A logistic regression model to predict heart disease from 13 features

This model predicts the presence or absence of heart disease based on 13 medical and demographic features. It uses data from https://www.kaggle.com/ronitf/heart-disease-uci, which was published by researchers in de-identified form. There are 296 cases total, of which 136 indicate heart disease.

The notebook was built with Python 3.8.10. The YAML file lists the required packages and may be used to create a conda environment with these packages.

For questions or comments, please contact me at dvschwab@protonmail.com.

## Summary

The notebook fits 3 versions of a logistic regression model: (1) a full model with all cases and features; (2) a reduced model containing only the 9 features with the greatest predictive value (as determined by the strength of relationship with the target); and (3) a reduced model with 20 outliers trimmed. All perform well, with a median accuracy of 83% for the test data. None of the models are prone to false positives, although each model does slightly better at predicting the absence of heart disease than its presence. However, the cross-validation scores show sample dependence for each model, and the relatively small number of cases per model limit the use of this analysis for patient diagnosis.
