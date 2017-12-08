
# coding: utf-8

# * reasoning : on input and on the model
# * calculate variable importances, try LR first

# # Prediction
# Predict whether a  particular company will default within 5  years, given its financial statement data (numeric only)

# # What is the input that we will get here?
# Most probably they will evaluate us on randomly selected values from a year and we have to give our bankruptcy predictions in probabilities of company going bankrupt till the 5th year.
# Also we need to justify the feature importances of why we say they would. Like because so and so ratio's trend is this which is a marker of bankruptcy. So it can even be like top 5 trends which are good marker for bankruptcy.

# # What is the nature of data?

# We will do a mix of all the 5 years for train and test.

# Correlation b/w Attr1 and Attr48 is significant so we are thinking of removing netprofit(Attr1) as it doesn't hold much value
# EBITDA is anyways very similar to netprofit ratio

# try single model for concateneted dataset or different model on 5 categories separately

# try correlation with 64 features

# There is an imbalance in the companies got bankrupt vs stable ones

# trends which have holded over the 5 yearsm

# find the outliers and see if most of them are labeled bankrupt

# ensure train and test have similar balances

# create API, and divide into modules, define interfaces

# Log transform
# https://blogs.sas.com/content/iml/2011/04/27/log-transformations-how-to-handle-negative-data-values.html

# # Findings
# Attr60, 45, 37, 21 , we have decided to drop these columns. 37 because it has very large missing values. And others because they have missing values + very low feature importance.

# We also found out that Attr27 was PITA because its one of the top 8 features but also one of the top 3 features with most missing values. Just blindly doing a dropna resulted in 30 % drop of bankruptcy class data point annhilation. WHICH WE CAN't ACCEPT IN OUR RIGHT MINDS with such significant class imbalances already in place.

# Hence this calls for clever missing data handling.

# After playing with imputation with median we thought of keeping 37 and 21 removed but fill the rest of dropped columns with their medians same as 27. Our reasoning is that they have same no. of missing values and since we are calculating importances using RandomForestClassifier, it may randomly select unselect features.
