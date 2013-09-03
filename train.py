__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'
'''
Train classifiers, perform cross validation, and make predictions
'''

import time
from datetime import datetime
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn.externals import joblib

def cross_validate(mtxTrn,mtxTarget,clf,folds=5,SEED=42,test_size=.15):
    mean_rmse = 0.0
    SEED = SEED *  time.localtime().tm_sec
    print "CV started at:", datetime.now().strftime("%d-%m-%y %H:%M")
    print "========================================="
    #scores = cross_validation.cross_val_score(clf, mtxTrn, mtxTarget, cv=folds, random_state=i*SEED+1, test_size=test_size, scoring=)
    for i in range(folds):
        #For each fold, create a test set (test_holdout) by randomly holding out X% of the data as CV set, where X is test_size (default .15)
        train_cv, test_cv, y_target, y_true = cross_validation.train_test_split(mtxTrn, mtxTarget, test_size=test_size, random_state=i*SEED+1)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        #Train model and make predictions on CV set
        clf.fit(train_cv, y_target)
        preds = clf.predict(test_cv)

        #For this CV fold, measure the error (distance between the predictions and the actual targets)
        rmse = metrics.mean_squared_error(y_true, preds)**(.5)
        print "RMSE (fold %d/%d): %f" % (i + 1, folds, rmse)
        mean_rmse += rmse
    print "========================================="
    print "Total mean RMSE score: %f" % (mean_rmse/folds)
    print "========================================="
    print "CV completed at:", datetime.now().strftime("%d-%m-%y %H:%M")

def cross_validate_using_benchmark(benchmark_name, dfTrn, mtxTrn,mtxTarget,folds=5,SEED=42,test_size=.15):
    mean_rmse = 0.0
    SEED = SEED *  time.localtime().tm_sec
    print "CV w/ avgs started at:", datetime.now().strftime("%d-%m-%y %H:%M")
    print "========================================="
    #scores = cross_validation.cross_val_score(clf, mtxTrn, mtxTarget, cv=folds, random_state=i*SEED+1, test_size=test_size, scoring)
    for i in range(folds):
        #For each fold, create a test set (test_holdout) by randomly holding out X% of the data as CV set, where X is test_size (default .15)
        train_cv, test_cv, y_target, y_true = cross_validation.train_test_split(mtxTrn, mtxTarget, test_size=test_size, random_state=SEED*i+10)

        #Calc benchmarks and use them to make a prediction
        benchmark_preds = 0
        if benchmark_name =='global_mean':
            #find global rev star mean for the training set sent
            benchmark = dfTrn[0].rev_stars.mean()
            benchmark_preds = [benchmark for x in test_cv]
        if benchmark_name =='business_mean':
            #uses business mean for each review
            benchmark_preds = [x[10] for x in test_cv]
        if benchmark_name =='usr_mean':
            #find user avg stars mean
            benchmark_preds = [x[5] for x in test_cv]
        if benchmark_name =='user_avg_stars_wtd_mean':
            #find user avg stars mean
            pass
        if benchmark_name =='3.5':
            #find user avg stars mean
            benchmark_preds = [3.5 for x in test_cv]
        print 'Using benchmark %s:' % (benchmark_name)

        #For this CV fold, measure the error (distance between the predictions and the actual targets)
        rmse = metrics.mean_squared_error(y_true, benchmark_preds)**(.5)
        print "RMSE (fold %d/%d): %f" % (i + 1, folds, rmse)
        mean_rmse += rmse
    print "========================================="
    print "Total mean RMSE score: %f" % (mean_rmse/folds)
    print "========================================="
    print "CV completed at:", datetime.now().strftime("%d-%m-%y %H:%M")
    print benchmark_preds
    return benchmark_preds

def predict(mtxTrn,mtxTarget,mtxTest,dfTest,clf,clfname):
    #fit the classifier
    clf.fit(mtxTrn, mtxTarget)

    #make predictions on test data and store them in the test data frame
    dfTest['predictions_'+clfname] = [x for x in clf.predict(mtxTest)]

    #print "Coefs for",clfname,clf.coef_
    return dfTest,clf

def save_predictions(dfTest,clf_name,model_name,submission_no):
    timestamp = datetime.now().strftime("--%d-%m-%y_%H%M")
    filename = 'Submits/'+'Submission'+submission_no+timestamp+'--'+clf_name+'--'+model_name+'.csv'

    #-perform any manual predictions cleanup that may be necessary-#
    ##convert any predictions below 1.2 to 1.2's and any predictions above 4.8 to 4.8's
    dfTest['predictions_'+clf_name] = [x if x < 1.3 else x[0] for x in dfTest['predictions_'+clf_name]] #may require x[0]
    dfTest['predictions_'+clf_name] = [x if x > 4.85 else x for x in dfTest['predictions_'+clf_name]]

    #save predictions
    dfTest.ix[:,['review_id','predictions_'+clf_name]].to_csv(filename,cols=['review_id','stars'], index=False)
    print 'Submission file saved as ',filename
    
def save_predictions_benchmark(dfTest_Benchmark,benchmark_name,submission_no):
    timestamp = datetime.now().strftime("--%d-%m-%y_%H%M")
    filename = 'Submissions/'+'Submission'+submission_no+timestamp+'--'+benchmark_name+'.csv'

    #save predictions
    dfTest_Benchmark.ix[:,['RecommendationId','benchmark_'+benchmark_name]].to_csv(filename,cols=['RecommendationId','stars'], index=False)
    print 'Submission file saved as ',filename

def save_model(clf,clfname):
    timestamp = datetime.now().strftime("%d-%m-%y_%H%M")
    filename = 'Models/'+timestamp+'--'+clfname+'.joblib.pk1'
    joblib.dump(clf, filename, compress=9)
    print 'Model saved as ',filename

def load_model(filename):
    return joblib.load(filename)

