# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:19:35 2017

@author: Juilee Rege
"""

import utils
import pandas as pd
from datetime import timedelta
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT
deliverables_path = '../deliverables/'
def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''

    indx_date = ''
    
    dead = mortality[['patient_id','timestamp']]
    dead['timestamp']=dead['timestamp'].apply(pd.to_datetime)
    dead_idx=dead.copy()
    dead_idx['timestamp'] = dead['timestamp']-timedelta(days=30)
    
    alive = events[(~events.patient_id.isin(dead.patient_id))] 
    alive = alive[['patient_id','timestamp']]
    alive['timestamp']=alive['timestamp'].apply(pd.to_datetime)
    alive_idx=alive.copy()
    alive_idx = alive.groupby('patient_id').timestamp.max()
    alive_idx = alive_idx.reset_index()
    
    indx_date = pd.concat([alive_idx,dead_idx])
    indx_date = indx_date.rename(columns = {'timestamp':'indx_date'})
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
   
    return indx_date

def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    filtered_events = ''
    
    filtered = pd.merge(indx_date, events, how='left', on=['patient_id'])
    filtered['timestamp']=filtered['timestamp'].apply(pd.to_datetime)
    filtered['indx_date']=filtered['indx_date'].apply(pd.to_datetime)
    filter1=filtered.copy()
    filter1['timestamp1'] = filtered['indx_date']-timedelta(days=2000)
    
    filter1 = filter1.loc[(filter1['timestamp'] <= filter1['indx_date']) & (filter1['timestamp'] >= filter1['timestamp1'])]
    filtered_events = filter1[['patient_id','event_id','value']].copy()
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    
    return filtered_events

def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    aggregated_events = ''
    
    joined = pd.merge(filtered_events_df, feature_map_df, how='left', on='event_id')
    filtered_events_not_null = joined.dropna()
       
    aggr = filtered_events_not_null.loc[filtered_events_not_null['event_id'].str.contains('D')]
    aggr = aggr[["patient_id","idx","value"]]
    aggr = aggr.groupby(["patient_id","idx"]).value.sum()#.groupby("idx")[[0]].sum().reset_index()
    aggr = aggr.reset_index()
       
    counted = filtered_events_not_null.loc[filtered_events_not_null['event_id'].str.contains('L')]
    counted = counted[["patient_id","idx","value"]]
    counted = counted.groupby(["patient_id","idx"]).value.count()#.groupby("idx")[[0]].sum().reset_index()
    counted = counted.reset_index()
    
    join1 = pd.concat([aggr,counted],axis=0) 
    max_df = join1.groupby('idx').value.max()
    max_df = max_df.reset_index()
    max_df = max_df.rename(columns = {'value':'max_value'})
    aggregated_events = pd.merge(join1, max_df,how='left',on=['idx'])
    aggregated_events['final_value'] = aggregated_events['value'] / aggregated_events['max_value']
    
    aggregated_events = aggregated_events[["patient_id","idx","final_value"]]
    aggregated_events = aggregated_events.rename(columns = {'idx':'feature_id','final_value':'feature_value'})
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    
    mortality_list = {}
    for i,j in mortality.iterrows():
        mortality_list[j['patient_id']] = j['label']
        
        
    patient_features_dict = {}
    for i, j in aggregated_events.iterrows():
        if j['patient_id'] not in patient_features_dict.keys():
            patient_features_dict[j['patient_id']] = []
        patient_features_dict[j['patient_id']].append((j['feature_id'], j['feature_value']))

    return patient_features_dict, mortality_list
    
def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    deliverable1.write("");
       
    for key, items in patient_features.items():
        if key in mortality.keys():
            string = str(float(mortality[key])) + " " + ' '.join(("%d:%f" % (fid, float(fvalue)) for fid, fvalue in sorted(items))) + " "
        else:
            string = "0.0 " + ' '.join(("%d:%f" % (fid, float(fvalue)) for fid, fvalue in sorted(items))) + " "
        string += '\n'
        deliverable1.write(string)

    for key, items in patient_features.items():
        if key in mortality.keys():
            string = str(int(key)) + " " + str(mortality[key]) + " " + ' '.join(("%d:%f" % (fid, float(fvalue)) for fid, fvalue in sorted(items))) + " "
        else:
            string = str(int(key)) + " 0.0 " + ' '.join(("%d:%f" % (fid, float(fvalue)) for fid, fvalue in sorted(items))) + " "
        string += '\n'
        deliverable2.write(string)

    deliverable1.close()
    deliverable2.close()

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()