__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'
'''
Calculate and extract features for ML
-This module is focused on calc and extraction of features for use in ML
'''

from sklearn.feature_extraction import DictVectorizer
from sklearn import  preprocessing
from scipy.sparse import coo_matrix, hstack, vstack
import numpy as np
import pandas as pd
from datetime import datetime

def handcraft(dfTrn_All, dfTest_All, dfTest_Master,dfTest_MissingUsers,dfTopCats,dfTopCatsBusAvg,dfAllBus_WithGroupedCats, dfBusNames): # dfTest_NoBusStars, dfTest_NoUsrStars,dfTest_NoBusUsrStars ,dfTest_NoUsr , dfTest_NoUsrBusStars,
    #----------------------------------------------------------
    #Add hand-crafted features
    #----------------------------------------------------------

    '''
    tempArray=[];j=0
    for i in range(0,len(dfTest_MissingUsers)):
        user_id = dfTest_MissingUsers.user_id[i]
        for j in tempArray:
            if j[0] == user_id:
                j[1] += 1.0
                j[2] += dfTest_MissingUsers.rev_stars[i]
        else:
            tempArray.append([user_id,1.0,dfTest_MissingUsers.rev_stars[i]])
    '''
    ##Calc avg review score for user id's in the training set that are also found in the test set
    tempDict={'user_id': [],'calc_user_rev_count':[],'calc_user_avg_stars':[]}
    for i in range(0,len(dfTest_MissingUsers)):
        user_id = dfTest_MissingUsers.user_id[i]
        if user_id in tempDict['user_id']:
            index = tempDict['user_id'].index(user_id)
            tempDict['calc_user_rev_count'][index] += 1.0
            tempDict['calc_user_avg_stars'][index] += dfTest_MissingUsers.rev_stars[i]
        else:
            tempDict['user_id'].append(user_id)
            tempDict['calc_user_rev_count'].append(1.0)
            tempDict['calc_user_avg_stars'].append(dfTest_MissingUsers.rev_stars[i])
    ####calc the avg
    for i in range(0,len(tempDict['user_id'])):
        tempDict['calc_user_avg_stars'][i] = tempDict['calc_user_avg_stars'][i] / tempDict['calc_user_rev_count'][i]
    ####Join tempDict on master
    dfTest_Master = dfTest_Master.merge(pd.DataFrame(tempDict),how='left',on='user_id')

    ##Total checkins by all yelpers at the business reviewed
    i=0;tempDict = {}
    for key in dfTrn_All.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    dfTrn_All['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    for key in dfTest_All.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    dfTest_All['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    '''
    for key in dfTest_NoBusStars.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    dfTest_NoBusStars['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    for key in dfTest_NoUsrStars.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    dfTest_NoUsrStars['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    for key in dfTest_NoBusUsrStars.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    dfTest_NoBusUsrStars['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    for key in dfTest_NoUsr.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    dfTest_NoUsr['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    for key in dfTest_NoUsrBusStars.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    dfTest_NoUsrBusStars['calc_total_checkins'] = pd.Series(tempDict)
    i=0;tempDict = {}
    '''
    for key in dfTest_Master.chk_checkin_info:
        total = 0
        #print key, type(key)
        if(type(key) != float):
            for key2 in key:
                total += key[key2]
        tempDict[i] = total
        i+=1
    dfTest_Master['calc_total_checkins'] = pd.Series(tempDict)
    del tempDict

    #Add avg rev stars for the categories of the business reviewed.  If >1 categories then average the averages together
    dfTrn_All['calc_cat_stars'] = 0.0
    dfTrn_All['calc_cat_count'] = 0.0
    j=0
    for row in dfTrn_All.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in dfTopCats.index.tolist():
                    dfTrn_All['calc_cat_stars'][j] += dfTopCats['cat_avg_stars'][i]
                    dfTrn_All['calc_cat_count'][j] += 1
        j+=1
    dfTrn_All['calc_cat_avg'] = dfTrn_All['calc_cat_stars'] / dfTrn_All['calc_cat_count']
    del dfTrn_All['calc_cat_stars']
    del dfTrn_All['calc_cat_count']

    dfTrn_All['calc_cat_stars_bus_avg'] = 0.0
    dfTrn_All['calc_cat_count_bus_avg'] = 0.0
    j=0
    for row in dfTrn_All.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in dfTopCatsBusAvg.index.tolist():
                    dfTrn_All['calc_cat_stars_bus_avg'][j] += dfTopCatsBusAvg['cat_avg_stars'][i]
                    dfTrn_All['calc_cat_count_bus_avg'][j] += 1
        j+=1
    dfTrn_All['calc_cat_avg_bus_avg'] = dfTrn_All['calc_cat_stars_bus_avg'] / dfTrn_All['calc_cat_count_bus_avg']
    dfTrn_All['calc_cat_avg'] = dfTrn_All['calc_cat_avg'].fillna(3.766723066)
    del dfTrn_All['calc_cat_stars_bus_avg']
    del dfTrn_All['calc_cat_count_bus_avg']

    dfTest_All['calc_cat_stars'] = 0.0
    dfTest_All['calc_cat_count'] = 0.0
    j=0
    for row in dfTest_All.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in dfTopCats.index.tolist():
                    dfTest_All['calc_cat_stars'][j] += dfTopCats['cat_avg_stars'][i]
                    dfTest_All['calc_cat_count'][j] += 1
        j+=1
    dfTest_All['calc_cat_avg'] = dfTest_All['calc_cat_stars'] / dfTest_All['calc_cat_count']
    dfTest_All['calc_cat_avg'] = dfTest_All['calc_cat_avg'].fillna(3.766723066)
    del dfTest_All['calc_cat_stars']
    del dfTest_All['calc_cat_count']

    dfTest_Master['calc_cat_stars'] = 0.0
    dfTest_Master['calc_cat_count'] = 0.0
    j=0
    for row in dfTest_Master.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in dfTopCats.index.tolist():
                    dfTest_Master['calc_cat_stars'][j] += dfTopCats['cat_avg_stars'][i]
                    dfTest_Master['calc_cat_count'][j] += 1
        j+=1
    dfTest_Master['calc_cat_avg'] = dfTest_Master['calc_cat_stars'] / dfTest_Master['calc_cat_count']
    del dfTest_Master['calc_cat_stars']
    del dfTest_Master['calc_cat_count']

    dfTest_Master['calc_cat_stars_bus_avg'] = 0.0
    dfTest_Master['calc_cat_count_bus_avg'] = 0.0
    j=0
    for row in dfTest_Master.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in dfTopCatsBusAvg.index.tolist():
                    dfTest_Master['calc_cat_stars_bus_avg'][j] += dfTopCatsBusAvg['cat_avg_stars'][i]
                    dfTest_Master['calc_cat_count_bus_avg'][j] += 1
        j+=1
    dfTest_Master['calc_cat_avg_bus_avg'] = dfTest_Master['calc_cat_stars_bus_avg'] / dfTest_Master['calc_cat_count_bus_avg']
    del dfTest_Master['calc_cat_stars_bus_avg']
    del dfTest_Master['calc_cat_count_bus_avg']

    #---Add the business names---#
    dfTrn_All = dfTrn_All.merge(dfBusNames[dfBusNames['bus_name_counts'] > 1],how='outer',on='bus_name')
    dfTest_Master = dfTest_Master.merge(dfBusNames,how='outer',on='bus_name')

    #---Add the grouped cat averages---#
    dfTrn_All = dfTrn_All.merge(dfAllBus_WithGroupedCats,how='left',on='business_id')
    dfTest_Master = dfTest_Master.merge(dfAllBus_WithGroupedCats,how='left',on='business_id')

    #Remove data fields used in calculations that are no longer needed
    del dfTrn_All['chk_checkin_info'];del dfTest_All['chk_checkin_info']
    '''
    del dfTest_NoBusStars['chk_checkin_info'];del dfTest_NoUsrStars['chk_checkin_info']
    del dfTest_NoBusUsrStars['chk_checkin_info'];del dfTest_NoUsr['chk_checkin_info']
    '''

    return dfTrn_All, dfTest_All,dfTest_Master  #return our beautiful homemade features     #dfTest_NoBusStars, dfTest_NoUsrStars, dfTest_NoBusUsrStars, dfTest_NoUsr, dfTest_NoUsrBusStars,

def vectorize(dfTrn_All,dfTest_All,dfTest_NoBusStars,dfTest_NoUsrStars,dfTest_NoBusUsrStars,dfTest_NoUsr,dfTest_NoUsrBusStars, feature):
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #For all models, vectorize a categorical feature using the DictVectorizer -- first fit it on both train and test sets to get all possible categories, then use it to transform each set into vectors
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vec = DictVectorizer().fit([{feature:value} for value in dfTrn_All.ix[:,feature].values])
    vecTrn = vec.transform([{feature:value} for value in dfTrn_All.ix[:,feature].values])
    vecTest = vec.transform([{feature:value} for value in dfTest_All.ix[:,feature].values])
    vecTest_NoBusStars = vec.transform([{feature:value} for value in dfTest_NoBusStars.ix[:,feature].values])
    vecTest_NoUsrStars = vec.transform([{feature:value} for value in dfTest_NoUsrStars.ix[:,feature].values])
    vecTest_NoBusUsrStars = vec.transform([{feature:value} for value in dfTest_NoBusUsrStars.ix[:,feature].values])
    vecTest_NoUsr = vec.transform([{feature:value} for value in dfTest_NoUsr.ix[:,feature].values])
    vecTest_NoUsrBusStars = vec.transform([{feature:value} for value in dfTest_NoUsrBusStars.ix[:,feature].values])

    return vecTrn, vecTest, vecTest_NoBusStars, vecTest_NoUsrStars,vecTest_NoBusUsrStars,vecTest_NoUsr,vecTest_NoUsrBusStars

def vectorizeMaster(dfTrn_All,dfTest_Master, feature):
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #For Master model only, vectorize a categorical feature using the DictVectorizer -- first fit it on both train and test sets to get all possible categories, then use it to transform each set into vectors
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vec = DictVectorizer().fit([{feature:value} for value in dfTrn_All.ix[:,feature].values])
    vecTrn = vec.transform([{feature:value} for value in dfTrn_All.ix[:,feature].values])
    vecTest_Master = vec.transform([{feature:value} for value in dfTest_Master.ix[:,feature].values])

    return vecTrn, vecTest_Master

def vectorize_buscategory(dfTest_Master, dfVec):
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    #Vectorize business categories into a hand made matrix (DictVectorizer does not work because the categories are nested and there can be many per record)
    #Send function each frame that you need to vectorize the business category in (dfTest_All,etc.)
    #------------------------------------------------------------------------------------------------------------------------------------------------------
    ##bus_categories -- create binary matrix
    listCats = []
    dfCats = dfVec.ix[:,['bus_categories']]
    j=0
    #make a complete list of all categories in the test set by extracting them from the nested lists
    for row in dfTest_Master.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                listCats.append(i)
        j+=1
    #Take the top 450 categories
    indexer = pd.Series(listCats).value_counts()[:450]
    #create new dataframes with one column for each category and initialize the values to 0
    for row in indexer.index.tolist():
        dfCats[row] = 0
    del dfCats['bus_categories']
    #Iterate through every record in each data set and if the category matches any of the columns, then set value in category data frame to 1
    j=0
    for row in dfVec.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in indexer.index.tolist():
                    dfCats[i][j] = 1
        j+=1
    del dfVec['bus_categories']
    return dfCats.as_matrix(), dfVec, indexer  #return our playful cats and the indexer (list of top 450 cats)

def standardize(dfTrn,dfTest,quant_features):
    #---------------------------------------------------------------------
    #Standardize list of quant features (remove mean and scale to unit variance)
    #---------------------------------------------------------------------
    scaler = preprocessing.StandardScaler()
    #dfTest[1].bus_review_count = dfTest[1].bus_review_count.astype(np.int32)
    mtxTrn= scaler.fit_transform(dfTrn.ix[:,quant_features].as_matrix())
    mtxTest = scaler.transform(dfTest.ix[:,quant_features].as_matrix())

    return mtxTrn, mtxTest #standard function return (see what I did there?)