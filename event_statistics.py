import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    avg_dead_event_count = 0.0
    max_dead_event_count = 0.0
    min_dead_event_count = 0.0
    avg_alive_event_count = 0.0
    max_alive_event_count = 0.0
    min_alive_event_count = 0.0
       
    dead = pd.merge(mortality, events, how='left', on=['patient_id'])
    total_dead_count = dead.groupby('patient_id').size()
    max_dead_event_count = total_dead_count.max()
    min_dead_event_count = total_dead_count.min()
    avg_dead_event_count = total_dead_count.mean()
    
    alive = events[(~events.patient_id.isin(dead.patient_id))]
    total_alive_count = alive.groupby('patient_id').size()
    max_alive_event_count = total_alive_count.max()
    min_alive_event_count = total_alive_count.min()
    avg_alive_event_count = total_alive_count.mean()
    
    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    avg_dead_encounter_count = 0.0
    max_dead_encounter_count = 0.0
    min_dead_encounter_count = 0.0 
    avg_alive_encounter_count = 0.0
    max_alive_encounter_count = 0.0
    min_alive_encounter_count = 0.0
    
    dead = pd.merge(mortality, events, how='left', on=['patient_id'])
    total_dead_count = dead.groupby('patient_id').timestamp_y.nunique()
    max_dead_encounter_count = total_dead_count.max()
    min_dead_encounter_count = total_dead_count.min()
    avg_dead_encounter_count = total_dead_count.mean()
    
    alive = events[(~events.patient_id.isin(dead.patient_id))]
    total_alive_count = alive.groupby('patient_id').timestamp.nunique()
    max_alive_encounter_count = total_alive_count.max()
    min_alive_encounter_count = total_alive_count.min()
    avg_alive_encounter_count = total_alive_count.mean()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    avg_dead_rec_len = 0.0
    max_dead_rec_len = 0.0
    min_dead_rec_len = 0.0
    avg_alive_rec_len = 0.0
    max_alive_rec_len = 0.0
    min_alive_rec_len = 0.0

    dead = pd.merge(mortality, events, how='left', on=['patient_id'])
    dead_dt = pd.to_datetime(dead.groupby('patient_id').timestamp_y.max()) .sub(pd.to_datetime(dead.groupby('patient_id').timestamp_y.min()))
    max_dead_rec_len = int(dead_dt.max().days)
    min_dead_rec_len = int(dead_dt.min().days)
    avg_dead_rec_len = float(max_dead_rec_len + min_dead_rec_len)/2
    
    alive = events[(~events.patient_id.isin(dead.patient_id))]
    alive_dt = pd.to_datetime(alive.groupby('patient_id').timestamp.max()) .sub(pd.to_datetime(alive.groupby('patient_id').timestamp.min()))
    max_alive_rec_len = int(alive_dt.max().days)
    min_alive_rec_len = int(alive_dt.min().days)
    avg_alive_rec_len = float(max_alive_rec_len + min_alive_rec_len)/2
    
    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    You may change the train_path variable to point to your train data directory.
    OTHER THAN THAT, DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following line to point the train_path variable to your train data directory
    train_path = '../data/train/'
    
    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
