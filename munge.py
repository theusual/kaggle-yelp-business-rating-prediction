__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'

'''
Data loading, cleaning, and merging
-This module is focused on getting the data into a usable form for analysis (ETL/munging).
'''

import utils
import pandas as pd
import numpy as np
from datetime import datetime
import gc

def load_data_frames():
    #--------------------------------------------------------------------
    #Load the training and test data into an array of PANDAS data frames
    #--------------------------------------------------------------------
    dataDirectory = "Data/"
    dataFilesTrn = ["yelp_training_set_review.json",
                    "yelp_training_set_business.json",
                    "yelp_training_set_user.json",
                    "yelp_training_set_checkin.json"]
    dataFilesTest = ["yelp_test_set_review.json",
                    "yelp_test_set_business.json",
                    "yelp_test_set_user.json",
                    "yelp_test_set_checkin.json"]
    dfTrn = []
    dfTest = []

    for file in dataFilesTrn:
        dfTrn.append( pd.DataFrame( utils.load_data_json(dataDirectory+file) ) )
    for file in dataFilesTest:
        dfTest.append( pd.DataFrame( utils.load_data_json(dataDirectory+file) ) )
    dfIdLookupTable = utils.load_data_csv_to_df(dataDirectory+'IdLookupTable.csv')
    return dfTrn, dfTest, dfIdLookupTable

def load_sent_score():
    #-------------------------------------------------------
    #Load pre-calculated sentiment scores for each review
    #-------------------------------------------------------
    dfTrnSent = pd.read_csv("Data/yelp_training_set_sent_score.csv",
                       names = ['sent_score','review_id'])
    dfTestSent = pd.read_csv("Data/yelp_test_set_sent_score.csv",
                        names = ['sent_score', 'review_id'])
    return dfTrnSent, dfTestSent

def data_cleaning(dfTrn,dfTest):
    #----------------------------------------------------------------------------------------------
    #Clean the data of inconsistencies, bad date fields, bad data types, nested columns, etc.
    #----------------------------------------------------------------------------------------------

    #Clean any bad data, usually by inserting global averages
    dfTrn[2][dfTrn[2].average_stars < 1] = dfTrn[2].average_stars.mean()

    #Clean bad characters
    dfTrn[2]['name'] = [x.encode("utf-8") if type(x) != float else x for x in dfTrn[2]['name']]
    dfTest[2]['name'] = [x.encode("utf-8") if type(x) != float else x for x in dfTest[2]['name']]

    #----Convert any data types------------

    #----Flatten any nested columns--------

    #----Data extractions------------
    #Extract zip code from full address
    dfTrn[1]['zip_code'] = [str(addr[-5:]) if 'AZ' not in addr[-5:] else 'Missing' for addr in dfTrn[1].full_address]
    dfTest[1]['zip_code'] = [str(addr[-5:]) if 'AZ' not in addr[-5:] else 'Missing' for addr in dfTest[1].full_address]

    #----Round or bin any continuous variables----
    ## Note: Binning is not needed for continuous variables when using RF or linear regression models, however binning can be useful
    ## for creating categorical variables

    #Round Longitude,Latitude
    dfTrn[1]['longitude_rounded2'] = [round(x,2) for x in dfTrn[1].longitude]
    dfTrn[1]['latitude_rounded2'] = [round(x,2) for x in dfTrn[1].latitude]
    dfTest[1]['longitude_rounded2'] = [round(x,2) for x in dfTest[1].longitude]
    dfTest[1]['latitude_rounded2'] = [round(x,2) for x in dfTest[1].latitude]

    #----delete any unused and/or redundant data from the frames----
    del dfTrn[0]['type'];del dfTest[0]['type']
    del dfTrn[0]['date'];del dfTrn[0]['text'];
    del dfTrn[1]['type'];del dfTest[1]['type']
    del dfTrn[1]['full_address'];del dfTest[1]['full_address']
    del dfTrn[1]['neighborhoods'];
    del dfTrn[1]['city'];del dfTest[1]['city']
    del dfTrn[1]['name'];del dfTest[1]['name']
    del dfTrn[2]['type'];del dfTest[2]['type']
    del dfTrn[3]['type'];del dfTest[3]['type']

    return dfTrn,dfTest #---return the fresh and clean data!---

def data_renaming(dfTrn,dfTest):
    #----------------------------------------------------------------------------------------------------------------
    #Data renaming as needed (to help with merges between tables that field names overlap, etc.)
    #----------------------------------------------------------------------------------------------------------------

    #rename all columns for clarity, except the keys
    dfTrn[0].columns = ['rev_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTrn[0]]
    dfTest[0].columns = ['rev_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTest[0]]
    dfTrn[1].columns = ['bus_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTrn[1]]
    dfTest[1].columns = ['bus_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTest[1]]
    dfTrn[2].columns = ['user_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTrn[2]]
    dfTest[2].columns = ['user_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTest[2]]
    dfTrn[3].columns = ['chk_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTrn[3]]
    dfTest[3].columns = ['chk_'+col if col not in ('business_id','user_id','review_id') else col for col in dfTest[3]]
    return dfTrn,dfTest  #---return the newly christened data!---

def data_compression(dfTrn,dfTest):
    #------------------------------------------------------------------------------------
    #Data type compression to save on overhead
    #------------------------------------------------------------------------------------

    #dfTrn[1].bus_review_count = dfTrn[1].bus_review_count.astype(np.int32)
    #dfTest[1].bus_review_count = dfTest[1].bus_review_count.astype(np.int32)
    #dfTrn[0].rev_stars = dfTrn[0].rev_stars.astype(np.int8)
    return dfTrn,dfTest

def load_combined_data_frames(dfTrn,dfTest):
    #------------------------------------------------------------------------------------------------------
    #combine training and test data for businesses, users, and checkins to create comprehensive data sets
    #------------------------------------------------------------------------------------------------------
    dfAll = ['','','','']
    for i in (1,2,3):
        dfAll[i] = dfTrn[i].append(dfTest[i])
    return dfAll

def data_merge(dfTrn,dfTest,dfAll, dfIdLookupTable):
    #-------------------------------------------------------------------
    #Data Merging - create joined data subsets for analysis and modeling
    #-------------------------------------------------------------------

    ## Pull RecommendationId from IdLookupTable for test reviews prior to other merges
    dfTest[0] = dfTest[0].merge(dfIdLookupTable,how='inner',on=['user_id','business_id'])

    ##Aggregated subsets
    ####Group all reviews that have a business with a star rating
    dfTest_Tot_BusStars = dfTest[0].merge(dfTrn[1],how='inner',on='business_id')
    del dfTest_Tot_BusStars['RecommendationId']
    dfTest_Tot_UsrStars = dfTest[0].merge(dfTrn[2],how='inner',on='user_id')
    del dfTest_Tot_UsrStars['RecommendationId']

    ##Create benchmark columns for merging with all other data subets
    global_rev_mean = dfTrn[0].rev_stars.mean()
    ####Business Mean -- Use business mean if known, use global review mean if not
    dfTest_Benchmark_BusMean = dfTest_Tot_BusStars.merge(dfTest[0],how='right',on=['business_id','user_id'])
    dfTest_Benchmark_BusMean['benchmark_bus_mean'] = dfTest_Benchmark_BusMean.bus_stars.fillna(global_rev_mean)
    dfTest_Benchmark_BusMean = dfTest_Benchmark_BusMean.ix[:,['RecommendationId','benchmark_bus_mean']]

    ####User Mean -- Use user mean if known, global review mean if not
    dfTest_Benchmark_UsrMean = dfTest_Tot_UsrStars.merge(dfTest[0],how='right',on=['business_id','user_id'])
    dfTest_Benchmark_UsrMean['benchmark_usr_mean'] = dfTest_Benchmark_UsrMean.user_average_stars.fillna(global_rev_mean)
    dfTest_Benchmark_UsrMean = dfTest_Benchmark_UsrMean.ix[:,['RecommendationId','benchmark_usr_mean']]

    ####Business and User mean -- When both use bus_stars, if neither use global review mean
    dfTest_Benchmark_BusUsrMean = dfTest_Tot_BusStars.merge(dfTest[0],how='right',on=['business_id','user_id'])
    dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'] = dfTest_Benchmark_BusUsrMean.bus_stars
    dfTest_Benchmark_BusUsrMean = dfTest_Benchmark_BusUsrMean.merge(dfTest_Tot_UsrStars,how='left',on=['business_id','user_id'])
    dfTest_Benchmark_BusUsrMean['benchmark_usr_mean'] = dfTest_Benchmark_BusUsrMean[np.isnan(dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'])]['user_average_stars']
    for x in range(0,len(dfTest_Benchmark_BusUsrMean)):
        if dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'][x] > 0:
            pass
        elif dfTest_Benchmark_BusUsrMean['benchmark_usr_mean'][x] > 0:
            dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'][x] = dfTest_Benchmark_BusUsrMean['benchmark_usr_mean'][x]
        else:
            dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'][x] = global_rev_mean
    #dfTest_Benchmark_BusUsrMean = dfTest_Benchmark_BusUsrMean.ix[:,['RecommendationId','benchmark_bus_usr_mean']]
    
    ####Business and user mean, when both use greater review count, if neither use global review mean
    dfTest_Benchmark_GrtrRevCountMean = dfTest[0].merge(dfTrn[1],how='inner', on='business_id')
    dfTest_Benchmark_GrtrRevCountMean = dfTest_Benchmark_GrtrRevCountMean.merge(dfTrn[2],how='inner', on='user_id')
    dfTest_Benchmark_GrtrRevCountMean = dfTest_Benchmark_GrtrRevCountMean.merge(dfAll[3],how='left', on='business_id')
    dfTemp = dfTest_Benchmark_GrtrRevCountMean[dfTest_Benchmark_GrtrRevCountMean.bus_review_count >= dfTest_Benchmark_GrtrRevCountMean.user_review_count]
    dfTemp['benchmark_grtr_rev_count_mean'] = dfTest_Benchmark_GrtrRevCountMean.bus_stars
    dfTemp2 = dfTest_Benchmark_GrtrRevCountMean[dfTest_Benchmark_GrtrRevCountMean.bus_review_count < dfTest_Benchmark_GrtrRevCountMean.user_review_count]
    dfTemp2['benchmark_grtr_rev_count_mean'] = dfTest_Benchmark_GrtrRevCountMean.user_average_stars
    dfTemp = dfTemp.append(dfTemp2)
    dfTest_Benchmark_BusUsrMean['benchmark_grtr_rev_count_mean'] = dfTemp['benchmark_grtr_rev_count_mean']
    dfTest_Benchmark_GrtrRevCountMean = dfTest_Benchmark_BusUsrMean
    for x in range(0,len(dfTest_Benchmark_GrtrRevCountMean)):
        if dfTest_Benchmark_GrtrRevCountMean['benchmark_grtr_rev_count_mean'][x] > 0:
            pass
        else:
            dfTest_Benchmark_GrtrRevCountMean['benchmark_grtr_rev_count_mean'][x] = dfTest_Benchmark_BusUsrMean['benchmark_bus_usr_mean'][x]

    ##Create master test set
    dfTest_Master = dfTest[0].merge(dfAll[1],how='left', on='business_id')
    dfTest_Master = dfTest_Master.merge(dfAll[2],how='left', on='user_id')
    dfTest_Master = dfTest_Master.merge(dfAll[3],how='left', on='business_id')

    ##Create set of reviews in training set that match user IDs in test review set but are not contained in the training user set
    dfTest_MissingUsers = dfTest[0].merge(dfTrn[2],how='left', on='user_id');del dfTest_MissingUsers['business_id']
    dfTest_MissingUsers = dfTest_MissingUsers[np.isnan(dfTest_MissingUsers['user_average_stars'])]
    dfTest_MissingUsers = dfTest_MissingUsers.merge(dfTrn[0],on='user_id',how='inner')
    del dfTest_MissingUsers['user_average_stars'];del dfTest_MissingUsers['user_name'];del dfTest_MissingUsers['user_review_count'];del dfTest_MissingUsers['user_votes']

    #dfTest[2].merge(dfTrn[0],on='user_id',how='inner')

    ## Create _All data subset - has business references with a star rating AND user references with avg stars
    dfTrn_All = dfTrn[0].merge(dfTrn[1],how='inner', on='business_id')
    dfTrn_All = dfTrn_All.merge(dfTrn[2],how='inner', on='user_id')
    dfTrn_All = dfTrn_All.merge(dfAll[3],how='left', on='business_id')

    dfTest_All = dfTest[0].merge(dfTrn[1],how='inner', on='business_id')
    dfTest_All = dfTest_All.merge(dfTrn[2],how='inner', on='user_id')
    dfTest_All = dfTest_All.merge(dfAll[3],how='left', on='business_id')

    ## Create _BusStars data subset - missing business references with a star rating but has user references with avg stars
    dfTest_BusStars = dfTest[0].merge(dfTest[1],how='inner', on='business_id')
    dfTest_BusStars = dfTest_BusStars.merge(dfTrn[2],how='inner', on='user_id')
    dfTest_BusStars = dfTest_BusStars.merge(dfAll[3],how='left', on='business_id')
    
    ## Create _NoUsrStars data subset - missing user references with avg stars, but has bus references with a star rating
    dfTest_NoUsrStars = dfTest[0].merge(dfTrn[1],how='inner', on='business_id')
    dfTest_NoUsrStars = dfTest_NoUsrStars.merge(dfTest[2],how='inner', on='user_id')
    dfTest_NoUsrStars = dfTest_NoUsrStars.merge(dfAll[3],how='left', on='business_id')

    ## Create _NoBusUsrStars data subset - missing user references with avg stars AND bus references with a star rating
    dfTest_NoBusUsrStars = dfTest[0].merge(dfTest[1],how='inner', on='business_id')
    dfTest_NoBusUsrStars = dfTest_NoBusUsrStars.merge(dfTest[2],how='inner', on='user_id')
    dfTest_NoBusUsrStars = dfTest_NoBusUsrStars.merge(dfAll[3],how='left', on='business_id')

    ## Create _NoUsr data subset - completely missing user references but containing bus_stars
    dfTest_NoUsr = dfTest[0].merge(dfAll[2],how='left', on='user_id')
    #####take records with a null references for user_average_stars then from that resulting df take records with null references for user_review_count
    dfTest_NoUsr = dfTest_NoUsr[np.isnan(dfTest_NoUsr['user_average_stars'])][np.isnan(dfTest_NoUsr['user_review_count'])]
    dfTest_NoUsr = dfTest_NoUsr.merge(dfTrn[1],how='inner', on='business_id')
    dfTest_NoUsr = dfTest_NoUsr.merge(dfAll[3],how='left', on='business_id')
    del dfTest_NoUsr['user_average_stars'];del dfTest_NoUsr['user_review_count']

    ####training data for subset
    dfTrn_NoUsr = dfTrn[0].merge(dfTrn[1],how='inner', on='business_id')
    dfTrn_NoUsr = dfTrn_NoUsr.merge(dfTrn[2],how='left', on='user_id')
    #take records with a null reference for user_avg_stars
    dfTrn_NoUsr = dfTrn_NoUsr[np.isnan(dfTrn_NoUsr['user_average_stars'])]
    dfTrn_NoUsr = dfTrn_NoUsr.merge(dfAll[3],how='left', on='business_id')
    del dfTrn_NoUsr['user_average_stars'];del dfTrn_NoUsr['user_review_count']

    ## Create _NoUsrBusStars data subset - completely missing user references AND missing bus_stars (only features will be from other bus info)
    dfTest_NoUsrBusStars = dfTest[0].merge(dfAll[2],how='left', on='user_id')
    #####take records with a null reference for user_average_stars then from that resulting df take records with null references for user_review_count
    dfTest_NoUsrBusStars = dfTest_NoUsrBusStars[np.isnan(dfTest_NoUsrBusStars['user_average_stars'])][np.isnan(dfTest_NoUsrBusStars['user_review_count'])]
    dfTest_NoUsrBusStars = dfTest_NoUsrBusStars.merge(dfAll[1],how='left', on='business_id')
    #####take records with a null reference for bus_stars
    dfTest_NoUsrBusStars = dfTest_NoUsrBusStars[np.isnan(dfTest_NoUsrBusStars['bus_stars'])]
    dfTest_NoUsrBusStars = dfTest_NoUsrBusStars.merge(dfAll[3],how='left', on='business_id')
    del dfTest_NoUsrBusStars['bus_stars']; del dfTest_NoUsrBusStars['user_average_stars'];del dfTest_NoUsrBusStars['user_review_count']

    return dfTrn_All,dfTest_All,dfTest_BusStars,dfTest_NoUsrStars,dfTest_NoBusUsrStars,dfTest_NoUsr,dfTest_NoUsrBusStars,dfTest_Tot_BusStars,dfTest_Benchmark_BusMean,dfTest_Benchmark_UsrMean,dfTest_Benchmark_BusUsrMean, dfTest_Master,dfTest_MissingUsers  #---return our new babies---#