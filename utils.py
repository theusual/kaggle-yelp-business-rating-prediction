__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'

'''
Generic utility functions useful in data-mining projects
'''

import json
import csv
import gc
from pandas import read_csv

def load_data_json(file_path):
    #import JSON data into a dict
    return [json.loads(line) for line in open(file_path)]

def load_data_csv(file_path):
    #import CSV into a list
    csv_reader = csv.reader(open(file_path),delimiter=',')
    return [line for line in csv_reader]

def load_data_csv_to_df(file_path):
    #import CSV into a dataframe
    return read_csv(file_path)

def data_garbage_collection(dfTrn,dfTest,dfAll):
    # Clean up unused frames:
    dfTrn[0] = '';dfTrn[2] = '';
    dfTest[0] = '';dfTest[2] = '';
    dfAll[1] = ''

    #garbage collection on memory
    gc.collect();
    return dfTrn,dfTest,dfAll
