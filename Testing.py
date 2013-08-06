#----------------------------------------------------------------------------------------------------------------
import sys
import json
import csv
import time
import gc
#import memory_profiler as mprof
from datetime import datetime
import pylab as pl

#math modules
import numpy as np
from numpy import genfromtxt, savetxt
from pandas import DataFrame, Series
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, hstack, vstack

#ML modules
from sklearn import (metrics, cross_validation, linear_model, preprocessing, ensemble)
from sklearn.feature_extraction import DictVectorizer


#def main():

#-------------------
#Data Loading
#-------------------

# <editor-fold desc="Description">
#JSON TRAINING Data -- load into separate data frames -------------
# business object
pathTrnBus = "Data/yelp_training_set_business.json";
dictTrnBus = [json.loads(line) for line in open(pathTrnBus)];frmTrnBus = DataFrame(dictTrnBus);del dictTrnBus;

#user objects
pathTrnUser = "Data/yelp_training_set_user.json";
dictTrnUser = [json.loads(line) for line in open(pathTrnUser)];frmTrnUser = DataFrame(dictTrnUser);del dictTrnUser;

#review objects
pathTrnRev = "Data/yelp_training_set_review.json";
dictTrnRev = [json.loads(line) for line in open(pathTrnRev)];frmTrnRev = DataFrame(dictTrnRev);del dictTrnRev;

#check ins objects
pathTrnChk = "Data/yelp_training_set_checkin.json";
dictTrnChk = [json.loads(line) for line in open(pathTrnChk)];frmTrnChk = DataFrame(dictTrnChk);del dictTrnChk;
#-------------------------

#JSON TEST data-----------------
# business object
pathTestBus = "Data/yelp_test_set_business.json";
dictTestBus = [json.loads(line) for line in open(pathTestBus)];frmTestBus = DataFrame(dictTestBus);del dictTestBus;

#user objects
pathTestUser = "Data/yelp_test_set_user.json";
dictTestUser = [json.loads(line) for line in open(pathTestUser)];frmTestUser = DataFrame(dictTestUser);del dictTestUser;

#review objects
pathTestRev = "Data/yelp_test_set_review.json";
dictTestRev = [json.loads(line) for line in open(pathTestRev)];frmTestRev = DataFrame(dictTestRev);del dictTestRev;

#check ins objects
pathTestChk = "Data/yelp_test_set_checkin.json";
dictTestChk = [json.loads(line) for line in open(pathTestChk)];frmTestChk = DataFrame(dictTestChk);del dictTestChk;
#-------------------------

#External data
#sentiment scores
frmTrnSentScores = pd.read_csv("academic_dataset/yelp_training_set_sent_score.csv",
                               names = ['sent_score','useful_votes', 'review_age', 'votes_per_day', 'review_stars','review_date', 'review_id'])
frmTestSentScores = pd.read_csv("Data/yelp_test_set_sent_score.csv",
                                names = ['sent_score', 'review_id'])
# </editor-fold>

#-----------------------------------------------------------------------------------------------
#append data frames to create comprehensive data frames
#-----------------------------------------------------------------------------------------------
# <editor-fold desc="Description">
frmAllBus = frmTrnBus.append(frmTestBus)
frmAllChk = frmTrnChk.append(frmTestChk)

#get rid of unused
del frmTrnChk;del frmTestChk;del frmTrnBus;del frmTestBus
# </editor-fold>

#-------------------
#Data Cleaning
#-------------------

# <editor-fold desc="Description">
#convert any data types
#Review Date - unicode data into datetime
frmTrnRev.date = [datetime.strptime(date, '%Y-%m-%d') for date in frmTrnRev.date]
frmTestRev.date = [datetime.strptime(date, '%Y-%m-%d') for date in frmTestRev.date]

#Flatten any nested columns
#business categories
#user votes
frmTrnUser['votes_cool'] = [rec['cool'] for rec in frmTrnUser.votes]
frmTrnUser['votes_funny'] = [rec['funny'] for rec in frmTrnUser.votes]
frmTrnUser['votes_useful'] = [rec['useful'] for rec in frmTrnUser.votes]

#review votes
frmTrnRev['votes_cool'] = [rec['cool'] for rec in frmTrnRev.votes]
frmTrnRev['votes_funny'] = [rec['funny'] for rec in frmTrnRev.votes]
frmTrnRev['votes_useful'] = [rec['useful'] for rec in frmTrnRev.votes]

#Other misc cleaning
#convert full_address to just a zip code
frmAllBus['zip_code'] = [str(addr[-5:]) if 'AZ' not in addr[-5:] else 'Missing' for addr in frmAllBus.full_address]
#frmTestBus['zip_code'] = [addr[-5:] if 'AZ' not in addr[-5:] else 0 for addr in frmTestBus.full_address]

#Round or bin any continuous variables
## Note: Binning is not needed for continuous variables when using RF or linear regression models, however binning can be useful
## for creating categorical variables

#Longitude
frmAllBus['longitude_rounded2'] = [round(x,2) for x in frmAllBus.longitude]
#frmTestBus['longitude_rounded2'] = [round(x,2) for x in frmTestBus.longitude]

#delete unused and/or redundant data from the objects
del frmAllBus['type']
del frmAllChk['type']
del frmTrnRev['type']
del frmTrnUser['type']
del frmAllBus['full_address']
del frmTrnRev['votes']
del frmTrnUser['votes']
del frmAllBus['neighborhoods']
del frmAllBus['state']
del frmTestRev['type']
del frmTestUser['type']
del frmTrnUser['votes_funny']
# <editor-fold desc="Description">
del frmTrnUser['votes_cool']
del frmAllBus['city']
del frmAllBus['name']
del frmTestUser['name']
del frmTrnUser['name']


#Remove unused calcs from sent scores
del frmTrnSentScores['review_age']
del frmTrnSentScores['useful_votes']
del frmTrnSentScores['votes_per_day']
del frmTrnSentScores['review_stars']
del frmTrnSentScores['review_date']
# </editor-fold>

#----------------------------------------------------------------------------------------------------------------
#Data renaming as needed (to help with merges between tables that field names overlap, etc.)
#----------------------------------------------------------------------------------------------------------------
# <editor-fold desc="Description">
#rename columns for clarity, except the keys

frmTrnRev.columns = ['rev_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTrnRev]
frmAllBus.columns = ['bus_'+col if col not in ('business_id','user_id','review_id') else col for col in frmAllBus]
frmTrnUser.columns = ['user_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTrnUser]
frmAllChk.columns = ['chk_'+col if col not in ('business_id','user_id','review_id') else col for col in frmAllChk]
frmTestRev.columns = ['rev_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTestRev]
frmTestUser.columns = ['user_'+col if col not in ('business_id','user_id','review_id') else col for col in frmTestUser]
# </editor-fold>

#------------------------------------------------------------------------------------
#Data type compression
#------------------------------------------------------------------------------------
#convert data types to data types that use less overhead

frmTrnSentScores.sent_score = frmTrnSentScores.sent_score.astype(np.int16)
frmAllBus.bus_review_count = frmAllBus.bus_review_count.astype(np.int32)
frmTrnRev.rev_stars = frmTrnRev.rev_stars.astype(np.int8)
frmTrnRev.rev_votes_cool = frmTrnRev.rev_votes_cool.astype(np.int16)
frmTrnRev.rev_votes_useful = frmTrnRev.rev_votes_useful.astype(np.int16)
frmTrnRev.rev_votes_funny = frmTrnRev.rev_votes_funny.astype(np.int16)
frmTestSentScores.sent_score = frmTestSentScores.sent_score.astype(np.int16)
frmTestRev.rev_stars = frmTestRev.rev_stars.astype(np.int8)
# </editor-fold>

#-------------------
#Data Merging
#-------------------
# <editor-fold desc="Description">
gc.collect();#print mprof.memory_usage()

## _All - data frames that include businesses, checkins, and users WITH user votes (uses frmTrnUser)
### Test data
frmTest_All = frmTestRev.merge(frmAllBus,how='inner', on='business_id')
frmTest_All = frmTest_All.merge(frmTrnUser,how='inner', on='user_id')
frmTest_All = frmTest_All.merge(frmAllChk,how='left', on='business_id')
frmTest_All = frmTest_All.merge(frmTestSentScores,how='left', on='review_id')
### Training data
frmTrn_All = frmTrnRev.merge(frmAllBus,how='inner', on='business_id')
frmTrn_All = frmTrn_All.merge(frmTrnUser,how='inner', on='user_id')
frmTrn_All = frmTrn_All.merge(frmAllChk,how='left', on='business_id')
frmTrn_All = frmTrn_All.merge(frmTrnSentScores,how='left', on='review_id')

'''
## _NoVotes -- data frames that include businesses and users from test data set (user records WITHOUT user votes)
### Test data
frmTest_NoVotes = frmTestRev.merge(frmAllBus,how='inner', on='business_id')
frmTest_NoVotes = frmTest_NoVotes.merge(frmTestUser,how='inner', on='user_id')
frmTest_NoVotes = frmTest_NoVotes.merge(frmAllChk,how='left', on='business_id')
frmTest_NoVotes= frmTest_NoVotes.merge(frmTestSentScores,how='left', on='review_id')

## _NoUser -- data frames that include businesses only, user record is missing
### Test data
frmAllUser = frmTrnUser.append(frmTestUser)
frmTest_NoUser= frmTestRev.merge(frmAllBus,how='inner', on='business_id')
frmTest_NoUser = frmTest_NoUser.merge(frmAllUser,how='left', on='user_id')
frmTest_NoUser = frmTest_NoUser.fillna(999)
frmTest_NoUser = frmTest_NoUser[frmTest_NoUser['user_average_stars'] == 999]
frmTest_NoUser = frmTest_NoUser.merge(frmAllChk,how='left', on='business_id')
frmTest_NoUser= frmTest_NoUser.merge(frmTestSentScores,how='left', on='review_id')
del frmTest_NoUser['user_votes_useful'];del frmTest_NoUser['user_average_stars'];del frmTest_NoUser['user_review_count']
'''

# Clean up unused frames:
del frmTrnRev; del frmTrnUser; del frmTestRev; del frmTestUser;del frmAllBus;

# </editor-fold>

#-------------------
#Data Calculations
#-------------------
# <editor-fold desc="Description">
gc.collect();#print mprof.memory_usage()

#Add calculated columns to the new merged document  (Calcs done AFTER merging in case they require fields across tables for calculation)
#Age
trainingDateCutoff = datetime.strptime('01-19-2013', '%m-%d-%Y')
testDateCutoff = datetime.strptime('03-12-2013', '%m-%d-%Y')
frmTrn_All['calc_rev_age'] = [(trainingDateCutoff - date).days  for date in frmTrn_All.rev_date]
frmTest_All['calc_rev_age'] = [(testDateCutoff - date).days  for date in frmTest_All.rev_date]
#frmTest_NoVotes['calc_rev_age'] = [(testDateCutoff - date).days  for date in frmTest_NoVotes.rev_date]
#frmTest_NoUser['calc_rev_age'] = [(testDateCutoff - date).days  for date in frmTest_NoUser.rev_date]


#Avg useful votes per day
frmTrn_All['calc_daily_avg_useful_votes'] = np.float64( frmTrn_All.rev_votes_useful[:] / np.float64(frmTrn_All.calc_rev_age[:]))

#Avg user useful votes per review
frmTrn_All['calc_user_avg_useful_votes'] = np.float64( frmTrn_All.user_votes_useful[:] / np.float64(frmTrn_All.user_review_count[:]))
frmTest_All['calc_user_avg_useful_votes'] = np.float64( frmTest_All.user_votes_useful[:] / np.float64(frmTest_All.user_review_count[:]))

#Length of review
frmTrn_All['calc_rev_length'] = [np.float64( len(rec) ) for rec in frmTrn_All.rev_text]
frmTest_All['calc_rev_length'] = [np.float64( len(rec) ) for rec in frmTest_All.rev_text]

#frmTest_NoVotes['calc_rev_length'] = [np.float64( len(rec) ) for rec in frmTest_NoVotes.rev_text]
#frmTest_NoUser['calc_rev_length'] = [np.float64( len(rec) ) for rec in frmTest_NoUser.rev_text]


#TotalCheckins
i=0;tempDict = {}
for key in frmTrn_All.chk_checkin_info:
    total = 0
    #print key, type(key)
    if(type(key) != float):
        for key2 in key:
            total += key[key2]
    tempDict[i] = total
    i+=1
frmTrn_All['calc_total_checkins'] = Series(tempDict)
i=0;tempDict = {}
for key in frmTest_All.chk_checkin_info:
    total = 0
    #print key, type(key)
    if(type(key) != float):
        for key2 in key:
            total += key[key2]
    tempDict[i] = total
    i+=1
frmTest_All['calc_total_checkins'] = Series(tempDict)

'''
i=0;tempDict = {}
for key in frmTest_NoVotes.chk_checkin_info:
    total = 0
    #print key, type(key)
    if(type(key) != float):
        for key2 in key:
            total += key[key2]
    tempDict[i] = total
    i+=1
frmTest_NoVotes['calc_total_checkins'] = Series(tempDict)
i=0;tempDict = {}
for key in frmTest_NoUser.chk_checkin_info:
    total = 0
    #print key, type(key)
    if(type(key) != float):
        for key2 in key:
            total += key[key2]
    tempDict[i] = total
    i+=1
frmTest_NoUser['calc_total_checkins'] = Series(tempDict)
'''

del tempDict

#Remove unused calc fields
del frmTrn_All['bus_latitude']
del frmTest_All['bus_latitude']
del frmTrn_All['bus_longitude']
del frmTest_All['bus_longitude']
del frmTrn_All['user_id']
del frmTest_All['user_id']
del frmTrn_All['rev_votes_cool']
del frmTrn_All['rev_votes_funny']
del frmTrn_All['rev_date']
del frmTest_All['rev_date']
del frmTrn_All['business_id']
del frmTest_All['business_id']
del frmTrn_All['chk_checkin_info']
del frmTest_All['chk_checkin_info']
del frmTrn_All['user_votes_useful']
del frmTest_All['user_votes_useful']

# </editor-fold>

#------------------
#Visualizations
#------------------
# <editor-fold desc="Description">
gc.collect();#print mprof.memory_usage()

reviewCounts = frmTrnBus['review_count'].value_counts()
reviewCounts[:10].plot(kind='barh',rot=1)
plt.show()

#plot a univariate relationship
X = frmTrn_All.ix[:10000,['calc_rev_age_scaled']].as_matrix()
Y = frmTrn_All.ix[:10000,['rev_votes_useful']].as_matrix()

clf = SGDRegressor(alpha=0.01, n_iter=100,shuffle=True)
clf.fit(X,Y)

pl.scatter(X,Y)
pl.plot(X, clf.predict(X), color='blue', linewidth=3)
pl.show()

   ##Another:
X = frmTrn_All.ix[:10000,['calc_total_checkins_scaled']].as_matrix()
Y = frmTrn_All.ix[:10000,['calc_daily_avg_useful_votes']].as_matrix()

clf = SGDRegressor(alpha=0.01, n_iter=100,shuffle=True)
clf.fit(X,Y)

pl.scatter(X,Y)
pl.plot(X, clf.predict(X), color='blue', linewidth=3)
pl.show()
# </editor-fold>

#------------------
#Machine Learning
#------------------

# <editor-fold desc="Description">
#Set # of records to process
trnRecords = 215879
#vecSplit = trnRecords / 2
#print vecSplit
testRecords = 13847

#-------------------------FEATURE SELECTION-------------------------------------------------------------------
##---Categorical features which need to be vectorized into a sparse matrix set of binary features ------------
#####bus_categories
frmTrn_All['bus_cat_1'] = ''
frmTrn_Cats = frmTrn_All.ix[:,['bus_categories']]
frmTest_Cats = frmTest_All.ix[:,['bus_categories']]
#frmTest_Cats = frmTest_NoVotes.ix[:,['bus_categories']]
#frmTest_Cats = frmTest_NoUser.ix[:,['bus_categories']]
j=0
for row in frmTrn_All.ix[:,['bus_categories']].values:
    for list in row:
        for i in list:
            frmTrn_All['bus_cat_1'][j] = i
    j+=1
indexer = frmTrn_All.bus_cat_1.value_counts()[:100]
gc.collect()
for row in indexer.index.tolist():
    frmTrn_Cats[row] = 0
    frmTest_Cats[row] = 0
j=0
del frmTrn_All['bus_cat_1'];del frmTrn_Cats['bus_categories'];del frmTest_Cats['bus_categories']
for row in frmTrn_All.ix[:,['bus_categories']].values:
    for list in row:
        for i in list:
            if i in indexer.index.tolist():
                #print i
                frmTrn_Cats[i][j] = 1
    j+=1
gc.collect()
trnCatsVec = frmTrn_Cats.as_matrix()
del frmTrn_Cats
gc.collect()
j=0
tempDf = frmTest_All
#tempDf = frmTest_NoVotes
#tempDf = frmTest_NoUser
for row in tempDf.ix[:,['bus_categories']].values:
    for list in row:
        for i in list:
            if i in indexer.index.tolist():
                #print i
                frmTest_Cats[i][j] = 1
    j+=1
testCatsVec = frmTest_Cats.as_matrix()
del frmTest_Cats;
gc.collect()

###fit the dictVectorizer's on data found in both train and test data, then use it to transform the train and test data into vectors
#####zip codes
catFeature = 'bus_zip_code'
vec = DictVectorizer().fit([{catFeature:value} for value in frmTrn_All.ix[:,catFeature].values])
trnZipVec = vec.transform([{catFeature:value} for value in frmTrn_All.ix[:trnRecords-1,catFeature].values])
testZipVec = vec.transform([{catFeature:value} for value in frmTest_All.ix[:testRecords-1,catFeature].values])
#testZipVec = vec.transform([{catFeature:value} for value in frmTest_NoVotes.ix[:testRecords-1,catFeature].values])
#testZipVec = vec.transform([{catFeature:value} for value in frmTest_NoUser.ix[:testRecords-1,catFeature].values])

#####bus_open
catFeature = 'bus_open'
vec = DictVectorizer().fit([{catFeature:value} for value in frmTrn_All.ix[:,catFeature].values])
trnBusOpenVec = vec.transform([{catFeature:value} for value in frmTrn_All.ix[:trnRecords-1,catFeature].values])
testBusOpenVec = vec.transform([{catFeature:value} for value in frmTest_All.ix[:testRecords-1,catFeature].values])
#testBusOpenVec = vec.transform([{catFeature:value} for value in frmTest_NoVotes.ix[:testRecords-1,catFeature].values])
#testBusOpenVec = vec.transform([{catFeature:value} for value in frmTest_NoUser.ix[:testRecords-1,catFeature].values])
del vec;

###------------Quantitative features which can be pulled directly from the dataframe without vectorizing---------------------------
####Not currently used features:  'calc_total_user_votes_scaled', 'user_votes_useful'
#quantFeatures = ['rev_stars_scaled','user_average_stars_scaled','sent_score_scaled','calc_total_checkins_scaled','calc_rev_length_scaled','calc_rev_age_scaled','user_votes_useful_scaled']
quantFeatures = ['rev_stars','user_average_stars','sent_score','calc_total_checkins','calc_rev_length','calc_rev_age','calc_user_avg_useful_votes','bus_review_count']
####Standardize/scale the quant variables
####For _All model:
scaler = preprocessing.StandardScaler()
train = scaler.fit_transform(frmTrn_All.ix[:trnRecords-1,quantFeatures].as_matrix())
test = scaler.transform(frmTest_All.ix[:testRecords-1,quantFeatures].as_matrix())

####For no votes model:
#quantFeatures = ['rev_stars','user_average_stars','sent_score','calc_total_checkins','calc_rev_length','calc_rev_age']
#scaler = preprocessing.StandardScaler()
#train = scaler.fit_transform(frmTrn_All.ix[:trnRecords-1,quantFeatures].as_matrix())
#test = scaler.transform(frmTest_NoVotes.ix[:testRecords-1,quantFeatures].as_matrix())

####For no user model:
#quantFeatures = ['rev_stars','sent_score','calc_total_checkins','calc_rev_length','calc_rev_age']
#scaler = preprocessing.StandardScaler()
#train = scaler.fit_transform(frmTrn_All.ix[:trnRecords-1,quantFeatures].as_matrix())
#test = scaler.transform(frmTest_NoVotes.ix[:testRecords-1,quantFeatures].as_matrix())

###-----------Combine all features into 1 matrix (sparse matrix of vectorized categorical features with matrix of scaled quant features)----------------
train = hstack([train,trnCatsVec,trnZipVec,trnBusOpenVec])
del trnCatsVec;del trnZipVec;del trnBusOpenVec;
test = hstack([test,testCatsVec,testZipVec,testBusOpenVec])
del testCatsVec;del testZipVec;del testBusOpenVec;

#select target
target = frmTrn_All.ix[:trnRecords-1,['rev_votes_useful']].as_matrix()

#select classifier--------
##  Common options:  ensemble -- RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
##                   linear_model -- SGDRegressor, Lasso
#clf = linear_model.LassoCV(cv=3)
#clf = linear_model.ElasticNet()
#clf = ensemble.RandomForestRegressor(n_estimators=50)
#clf = linear_model.SGDRegressor(alpha=0.0001, n_iter=2000,shuffle=True)

# === training & metrics === #
SEED = 21.
mean_auc = 0.0
n = 10  # repeat the CV procedure n times
for i in range(n):
    # for each iteration, randomly hold out 15% of the data as CV set
     train_cv, train_holdout, target_cv, test_holdout = cross_validation.train_test_split(
        train, test_size=.15, random_state=i*SEED)

    # if you want to perform feature selection / hyperparameter
    # optimization, this is where you want to do it

    # train model and make predictions
    clf.fit(train_cv, target_cv)
    preds = model.predict_proba(X_cv)[:, 1]

    # compute AUC metric for this CV fold
    fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
    mean_auc += roc_auc

print "Mean AUC: %f" % (mean_auc/n)

# ==== Make Predictions ==== #

#fit the classifier
#clf.fit(train, target)
clf.fit(train.toarray(), target)

#make predictions on test data
frmTest_All['predictions_LassoCV1'] = [x for x in clf.predict(test)]
#frmTest_NoVotes['predictions_SGD1'] = [x for x in clf.predict(test)]
#frmTest_User['predictions_SGD1'] = [x for x in clf.predict(test)]

#log coefficients generated for SGD
print clf.coef_

#compare latest classifier to other classifiers
frmTest_All[-50:]
#frmTest_NoVotes[-50:]
#frmTest_NoUser[-50:]

#convert negative votes to zeroes
frmTest_All['predictions_LassoCV1'] = [x if x > 0 else 0 for x in frmTest_All['predictions_LassoCV1']]
#frmTest_NoVotes['predictions_SGD1'] = [x if x > 0 else 0 for x in frmTest_NoVotes['predictions_SGD1']]
#frmTest_NoUser['predictions_SGD1'] = [x if x > 0 else 0 for x in frmTest_NoUser['predictions_SGD1']]


#save predictions
frmTest_All.ix[:,['review_id','predictions_LassoCV1']].to_csv('Data/submission2-AllOthers.csv',cols=['id','votes'], index=False)
#frmTest_NoVotes.ix[:,['review_id','predictions_SGD1']].to_csv('Data/submission1-NoVotes.csv',cols=['review_id','votes'], index=False)
#frmTest_NoUser.ix[:,['review_id','predictions_SGD1']].to_csv('Data/submission1-NoUser.csv',cols=['review_id','votes'], index=False)

#savetxt('Data/submission2.csv', frmTest_All.ix[:,['review_id','predictions_SGD1']], delimiter=',', fmt='%f')
# </editor-fold>


#-------------------
#misc functions
#-------------------

# <editor-fold desc="Description">
print frmTrnBus['review_count'][:10]
print frmTrnBus['review_count'].value_counts()
print frmTrnBus[frmTrnBus.city == 'Wittmann']
frmTrnBusByRevCount = frmTrnBus.groupby(['city','review_count'])

#Compare data in 2 columns
i=0
for x in frmAcadTrnRev.votes_useful_y:
    if  x != frmAcadTrnRev['votes_useful_x'][i]:
        print x
    i+=1
print 'done'
# </editor-fold>


if __name__ == '__main__':
    main()
