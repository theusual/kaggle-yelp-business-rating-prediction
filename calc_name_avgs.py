__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '08-24-2013'
'''
Extract average stars for businesses that share the same name, then dump it to a csv for import into the main analysis
'''

import numpy as np
import pandas as pd
from datetime import datetime

def calc_name_avgs(dfTrn):
    dfNames = pd.DataFrame(dfTrn[1]['bus_name'].value_counts())
    dfNames.columns = ['name_counts']
    dfNames['total_stars'] = 0.00
    j=0
    for row in dfTrn[1].ix[:,['bus_name']].values:
        dfNames['total_stars'][row[0]] += dfTrn[1]['bus_stars'][j]
        j+=1
    dfNames['bus_name_avg'] = dfNames['total_stars'] / dfNames['name_counts']

    #dump to csv
    file_path = "Data/bus_name_avg.csv"
    dfNames.to_csv(file_path)

    return