'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
    entropy = 0
    dis = df[df.columns[-1]].value_counts(normalize=True)
    for p in dis:
        if p > 0:
            entropy += -1 * p * np.log2(p)
    return entropy

'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df, attribute):
    entropy_of_attribute = 0
    total = len(df)
    res_col = df.columns[-1]
    if df[attribute].dtype == "float64":
        # Freedman-Diaconis rule
        iqr = df[attribute].quantile(0.75) - df[attribute].quantile(0.25)
        h = 2 * iqr * (total ** (-1/3))
        bins = (df[attribute].max() - df[attribute].min()) / h
        dis_vals = df[attribute].value_counts(bins=round(float(bins)), normalize=True)
    else:
        dis_vals = df[attribute].value_counts(normalize=True)
    vals = dis_vals.keys()
    results = df[res_col].value_counts().keys()

    for val in vals:
        entropy_val = 0
        if isinstance(val, pd.Interval):
            df_same_vals = df[(val.left < df[attribute]) & (df[attribute] <= val.right)]
        else:
            df_same_vals = df[df[attribute] == val]
        tot = len(df_same_vals)
        for result in results:
            res = len(df_same_vals[attribute][df_same_vals[res_col] == result])
            frac = res/tot
            if frac > 0:
                entropy_val = entropy_val + (-1 * frac) * np.log2(frac)

        entropy_of_attribute = entropy_of_attribute + (tot/total)*entropy_val

    return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df, attribute):
    total_entropy = get_entropy_of_dataset(df)
    entropy_of_attribute = get_entropy_of_attribute(df, attribute)
    information_gain = total_entropy - entropy_of_attribute

    return information_gain

''' Returns Attribute with highest info gain'''
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
    information_gains = {}

    attributes = list(df.columns)[:-1]
    temp = 0
    for attr in attributes:
        ig = get_information_gain(df, attr)
        information_gains[attr] = ig
        if ig > temp:
            selected_column = attr
            temp = ig

    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''

    return (information_gains, selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
