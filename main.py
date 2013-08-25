__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'

import utils
import munge
import features
import train

from datetime import datetime
import pandas as pd
from sklearn import (metrics, cross_validation, linear_model, ensemble, preprocessing, svm, neighbors,gaussian_process)
from scipy.sparse import coo_matrix, hstack, vstack

#--Set current submission number--#
submission_no = '99'

###def main():
#----------------------------------------#
#-------Data Loading/Cleaning/Munging----#
#----------------------------------------#

#--load training and test data frames--#
dfTrn, dfTest, dfIdLookupTable = munge.load_data_frames()

#--load avg stars for top categories --#
dfTopCats, dfTopCatsBusAvg = munge.load_category_avgs()

#--load avg stars for bus names --#
dfBusNames = munge.load_bus_name_avgs()

#--Data Cleaning--#
dfTrn,dfTest = munge.data_cleaning(dfTrn,dfTest)

#--Data renaming--#
dfTrn,dfTest = munge.data_renaming(dfTrn,dfTest)

#--Data type compression to save on overhead--#
dfTrn,dfTest = munge.data_compression(dfTrn,dfTest)

#--combine training and test data for businesses, users, and checkins [1,2,3] to create comprehensive data sets.  dfAll[0] is empty.--#
dfAll = munge.load_combined_data_frames(dfTrn,dfTest)

#--Data merging to create data subsets for training and testing--#
dfTrn_All,dfTest_All, dfTest_Master,dfTest_MissingUsers = munge.data_merge(dfTrn,dfTest,dfAll,dfIdLookupTable) #dfTest_NoBusStars,dfTest_NoUsrStars,dfTest_NoBusUsrStars,dfTest_NoUsr,dfTest_NoUsrBusStars,dfTest_Tot_BusStars,dfTest_Benchmark_BusMean,dfTest_Benchmark_UsrMean,dfTest_Benchmark_BusUsrMean,

#----------------------------------------#
#--------- Feature Selection-------------#`
#----------------------------------------#

#--Add handcrafted (calculated) features--#
dfTrn_All, dfTest_All, dfTest_Master = features.handcraft(dfTrn_All,dfTest_All, dfTest_Master,dfTest_MissingUsers, dfTopCats,dfTopCatsBusAvg) # dfTest_NoBusStars, dfTest_NoUsrStars, dfTest_NoBusUsrStars, dfTest_NoUsr, dfTest_NoUsrBusStars,

#--Remove outliers from sets prior to ML and vectorizing
##Remove review count outlier (airport?)
dfTrn_All = dfTrn_All[dfTrn_All['bus_review_count'] < 450]
##Remove checkin outliers
dfTrn_All = dfTrn_All[dfTrn_All['calc_total_checkins'] < 700]

#--Vectorize categorical features--#
#vecTrn_Zip, vecTest_All_Zip, vecTest_NoBusStars_Zip, vecTest_NoUsrStars_Zip, vecTest_NoBusUsrStars_Zip, vecTest_NoUsr_Zip, vecTest_NoUsrBusStars_Zip = features.vectorize(dfTrn_All,dfTest_All,dfTest_NoBusStars,dfTest_NoUsrStars,dfTest_NoBusUsrStars,dfTest_NoUsr,dfTest_NoUsrBusStars,'bus_zip_code')
#vecTrn_BusOpen, vecTest_All_BusOpen, vecTest_NoBusStars_BusOpen, vecTest_NoUsrStars_BusOpen, vecTest_NoBusUsrStars_BusOpen, vecTest_NoUsr_BusOpen, vecTest_NoUsrBusStars_BusOpen = features.vectorize(dfTrn_All,dfTest_All,dfTest_NoBusStars,dfTest_NoUsrStars,dfTest_NoBusUsrStars,dfTest_NoUsr,dfTest_NoUsrBusStars,'bus_open')
vecTrn_Zip, vecTest_Master_Zip  = features.vectorizeMaster(dfTrn_All,dfTest_Master,'bus_zip_code')
vecTrn_BusOpen,vecTest_Master_BusOpen  = features.vectorizeMaster(dfTrn_All,dfTest_Master,'bus_open')

#--Vectorize business categories (dictVectorizer does not work)--#
###Create a master list of bus categories used in the test reviews
#dfTest_Master = dfTest[0].merge(dfAll[1],on='business_id',how='inner')
vecTrn_Cats, dfTrn_All, topCats = features.vectorize_buscategory(dfTest_Master,dfTrn_All)
vecTest_All_Cats, dfTest_All , topCats = features.vectorize_buscategory(dfTest_Master,dfTest_All)
vecTest_NoBusStars_Cats,dfTest_NoBusStars,topCats = features.vectorize_buscategory(dfTest_Master,dfTest_NoBusStars)
vecTest_NoUsrStars_Cats,dfTest_NoUsrStars, topCats = features.vectorize_buscategory(dfTest_Master,dfTest_NoUsrStars)
vecTest_NoBusUsrStars_Cats,dfTest_NoBusUsrStars, topCats = features.vectorize_buscategory(dfTest_Master,dfTest_NoBusUsrStars)
vecTest_NoUsr_Cats,dfTest_NoUsr, topCats = features.vectorize_buscategory(dfTest_Master,dfTest_NoUsr)
vecTrn_NoUsr_Cats,dfTrn_NoUsr, topCats = features.vectorize_buscategory(dfTest_Master,dfTrn_NoUsr)
vecTest_NoUsrBusStars_Cats,dfTest_NoUsrBusStars, topCats = features.vectorize_buscategory(dfTest_Master,dfTest_NoUsrBusStars)

#----------------------------------------#
#--------- Machine Learning--------------#
#----------------------------------------#

#--select classifier--#
##  Common options:  ensemble -- RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
##                   linear_model -- SGDRegressor, Lasso

#clf = ensemble.RandomForestRegressor(n_estimators=50);  clfname='RFReg_50'
#clf = ensemble.ExtraTreesRegressor(n_estimators=30)  #n_jobs = -1 if running in a main() loop
#clf = linear_model.SGDRegressor(alpha=0.001, n_iter=800,shuffle=True); clfname='SGD_001_800'
#clf = gaussian_process.GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)   #doesn't work on large datasets
#clf = linear_model.Ridge();clf_name = 'RidgeReg'
#clf = linear_model.LinearRegression();clf_name = 'LinReg'
#clf = linear_model.ElasticNet()
#clf = linear_model.Lasso();clf_name = 'Lasso'
#clf = linear_model.LassoCV(cv=3);clf_name = 'LassoCV'
#clf = svm.SVR(kernel = 'linear',cache_size = 6000.0) #use .ravel(), kernel='rbf','linear'
#clf = svm.SVC(kernel = 'linear',cache_size = 6000.0) #use .ravel(), kernel='rbf','linear'
n_neighbors = 200; clf = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform', algorithm = 'kd_tree');clf_name='KNN_200' #use .toarray

#--------------Machine Learning (woohoo, we finally got to the good stuff)------------------------#
#quant_features = ['user_average_stars','user_review_count','calc_total_checkins','bus_stars','bus_review_count']
quant_features = ['bus_name_avg']
dfTrn_ML= dfTrn_BusNames_Grtr1; dfTest_ML= dfTest_BusNames_Grtr1;
mtxTrn,mtxTest = features.standardize(dfTrn_ML,dfTest_ML,quant_features)
#--Combine the standardized quant features and the vectorized categorical features--#
#mtxTrn = hstack([mtxTrn,vecTrn_BusOpen])  #vecTrn_BusOpen,vecTrn_Cats,vecTrn_Zip,
#mtxTest = hstack([mtxTest,vecTest_BusOpen]) #vecTest_Master_Cats,vecTest_Master_Zip,
#--Test without the vecZip and vecCats--#
#mtxTrn = hstack([mtxTrn,vecTrn_BusOpen])
#mtxTest = hstack([mtxTest,vecTest_Master_BusOpen])
#--select target--#
mtxTarget = dfTrn_ML.ix[:,['rev_stars']].as_matrix()

#--Use classifier for cross validation--#
train.cross_validate(mtxTrn,mtxTarget,clf,folds=10,SEED=42,test_size=.15)  #may require mtxTrn.toarray()

#--Use classifier for predictions--#
dfTest_ML, clf = train.predict(mtxTrn,mtxTarget,mtxTest,dfTest_ML,clf,clf_name) #may require mtxTest.toarray()

#--Save predictions to file--#
train.save_predictions(dfTest_ML,clf_name,'_BusName_ColdGrtr1',submission_no)

#---------End Machine Learning Section-------------#

#------------------------------Optional Steps----------------------------------#
#--Memory cleanup prior to running the memory intensive classifiers--#
dfTrn,dfTest,dfAll = utils.data_garbage_collection(dfTrn,dfTest,dfAll)

#--use a benchmark instead of a classifier--#
benchmark_preds = train.cross_validate_using_benchmark('3.5', dfTrn, dfTrn[0].merge(dfTrn[1],how='inner',on='business_id').as_matrix(),dfTrn[0].ix[:,['rev_stars']].as_matrix(),folds=3,SEED=42,test_size=.15)
benchmark_preds = train.cross_validate_using_benchmark('global_mean', dfTrn, dfTrn[0].merge(dfTrn[1],how='inner',on='business_id').as_matrix(),dfTrn[0].ix[:,['rev_stars']].as_matrix(),folds=3,SEED=42,test_size=.15)
benchmark_preds = train.cross_validate_using_benchmark('business_mean', dfTrn, dfTrn[0].merge(dfTrn[1],how='inner',on='business_id').as_matrix(),dfTrn[0].ix[:,['rev_stars']].as_matrix(),folds=3,SEED=42,test_size=.15)
benchmark_preds = train.cross_validate_using_benchmark('usr_mean', dfTrn, dfTrn[0].merge(dfTrn[2],how='inner',on='user_id').as_matrix(),dfTrn[0].merge(dfTrn[2],how='inner',on='user_id').ix[:,['rev_stars']].as_matrix(),folds=3,SEED=22,test_size=.15)

#--predict using a benchmark--#
train.save_predictions_benchmark(dfTest_Benchmark_BusMean,'bus_mean',submission_no)
train.save_predictions_benchmark(dfTest_Benchmark_UsrMean,'usr_mean',submission_no)
train.save_predictions_benchmark(dfTest_Benchmark_BusUsrMean,'bus_usr_mean',submission_no)

#--Save model to joblib file--#
train.save_model(clf,clf_name)

#--Save a dataframe to CSV--#
filename = 'Data/'+datetime.now().strftime("%d-%m-%y_%H%M")+'--FinalDataset--OldUserTest'+'.csv'
#del dfTest_Master['business_id'];del dfTest_Master['user_id'];
#dfTest_Master.ix[:,['RecommendationId','calc_user_avg_stars','calc_user_rev_count']].to_csv(filename, index=False)
dfTest_Old[2].to_csv(filename, index=False)

#--Save predictions to CSV--#
filename = 'Data/'+datetime.now().strftime("%d-%m-%y_%H%M")+'--Pred_ChkBus&Open_LinReg'+'.csv'
dfTest_Master['predictions_LinReg'] = [x[0] for x in dfTest_Master.predictions_LinReg]
dfTest_Master.ix[:,['RecommendationId','predictions_LinReg']].to_csv(filename, index=False)

#--Load model from joblib file--#
clf = train.load_model('Models/07-07-13_1247--SGD_001_1000.joblib.pk1')

if __name__ == '__main__':
    main()