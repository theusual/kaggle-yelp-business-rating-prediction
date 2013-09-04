kaggle-yelp-business-rating-prediction
======================================

Code for Kaggle Contest "RecSys2013: Yelp Business Rating Prediction" -- Predicting Business Ratings on Yelp.com Using Machine Learning
.

This was my third Kaggle contest and my final submission achieved an RMSE of 1.23107 which placed #7 out of ~250 teams.  I used the python code in this repository to load and munge the data and for modeling and cross-validation on certain subsets of the data that performed well with machine learning models.  The subsets were output to CSV's then joined together in Excel where factorization models were added for the remaining subsets not ran through ML, thus creating a comprehensive submission for the contest.

This code was mostly written stream of conscience during the weeks I worked on the contest, so it is messy and jumbled.  Also, it was intended to be run in the Ipython console in sections during exploration of the data, so it will not run as a stand-alone program.  Think of it more as a collection of useful code snippets for data analysis! 