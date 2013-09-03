__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '08-18-2013'
'''
Extract categories and calculate averages, then dump it to a csv for import into the main analysis
'''

import numpy as np
import pandas as pd
from datetime import datetime

def calc_categories(dfTrn_All):
    listCats = []
    dfCats = dfTrn_All.ix[:,['bus_categories']]
    j=0
    #make a complete list of all categories in the training set by extracting them from the nested lists
    for row in dfTrn_All.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                listCats.append(i)
        j+=1
    #Take the top 421 categories (cutoff at 15 reviews minimum)
    dfTopCats = pd.DataFrame(pd.Series(listCats).value_counts()[:421])
    dfTopCats.columns = ['cat_count']
    dfTopCats['cat_tot_stars'] = 0
    #Calc rev star average for each category:
    ##Iterate through every record in the training data set and if the category matches a top category, then add the review stars to that category's total stars
    j=0
    for row in dfTrn_All.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                if i in topCats.index.tolist():
                    dfTopCats['cat_tot_stars'][i] += dfTrn_All['rev_stars'][j]
        j+=1
    ##divide each category's total stars by its total rev count to derive the avg
    dfTopCats['cat_avg_stars'] = dfTopCats['cat_tot_stars'] / dfTopCats['cat_count'].astype(np.float16)
    del dfTopCats['cat_tot_stars']
    #dump to csv
    file_path = "Data/top_categories.csv"
    dfTopCats.to_csv(file_path)

    return

#--Different method using business avg's to derive category averages (therefore using population mean, thus it should be better than using sample mean above)--#
def calc_categories_using_bus_avg(dfAllBus):
    listCats = []
    #Remove businesses with < 3 reviews adn businesses with no business star ratings
    dfAllBus = dfAllBus[dfAllBus['bus_review_count'] > 3]
    dfAllBus = dfAllBus[dfAllBus['bus_stars'] > 0]
    dfAllBus = dfAllBus.reset_index(drop=True)
    #slice off the categories
    dfCats = dfAllBus.ix[:,['bus_categories']]
    #make a complete list of all categories by extracting them from the nested lists
    j=0
    for row in dfAllBus.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                listCats.append(i)
        j+=1
    dfTopCats = pd.DataFrame(pd.Series(listCats).unique())
    dfTopCats.columns = ['category']
    dfTopCats['cat_tot_stars'] = 0.0
    dfTopCats['cat_tot_count'] = 0.0
    dfTopCats = dfTopCats.set_index('category')
    #Calc rev star average for each category:
    ##Iterate through every business in the data set and add the review stars to that category's total stars
    j=0
    for row in dfAllBus.ix[:,['bus_categories']].values:
        for list in row:
            for i in list:
                dfTopCats['cat_tot_stars'][i] += dfAllBus['bus_stars'][j]
                dfTopCats['cat_tot_count'][i] += 1
        j+=1
    ##divide each category's total stars by its total rev count to derive the avg
    dfTopCats['cat_avg_stars'] = dfTopCats['cat_tot_stars'] / dfTopCats['cat_tot_count'].astype(np.float16)
    del dfTopCats['cat_tot_stars']
    #parse off any categories with less than 3 businesses
    dfTopCats = dfTopCats[dfTopCats['cat_tot_count'] > 2]
    #dump to csv
    file_path = "Data/top_categories_bus_avg.csv"
    dfTopCats.to_csv(file_path)

    return

#--Latest method using grouping all categories for a business together to create the feature (therefore [Restaurants, Mexican] is one category, not 2 like in the above methods)--#
def calc_group_categories(dfAllBus):
    #Remove businesses with < 5 reviews and businesses with no business star ratings AND businesses with no bus categories
    dfAllBus = dfAllBus[dfAllBus['bus_review_count'] > 4]
    dfAllBus = dfAllBus[dfAllBus['bus_stars'] > 0]
    #dfAllBus['bus_categories'] = [x if len(x) > 0 else 'MISSING' for x in dfAllBus.bus_categories]
    dfAllBus = dfAllBus.reset_index(drop=True)
    #slice off the categories
    dfTopCats = pd.DataFrame(dfAllBus['bus_categories'].value_counts())
    dfTopCats.columns = ['cat_counts']
    dfTopCats['total_stars'] = 0.00
    #Calc rev star average for each category:
    ##Iterate through every business in the data set and add the review stars to that category's total stars
    j=0
    for row in dfAllBus.ix[:,['bus_categories']].values:
        for cat in row:
            if len(cat) > 0:
                dfTopCats['total_stars'][cat] += dfAllBus['bus_stars'][j]
        j+=1
    ##divide each category's total stars by its total rev count to derive the avg
    dfTopCats['cat_avg_stars'] = dfTopCats['total_stars'] / dfTopCats['cat_counts'].astype(np.float16)
    del dfTopCats['total_stars']
    #dump to csv
    file_path = "Data/top_categories_grouped.csv"
    dfTopCats.to_csv(file_path)


