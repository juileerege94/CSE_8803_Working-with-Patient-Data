import utils
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this

    filepath = '../data/test/'
    mortality = pd.read_csv('../data/train/' + 'mortality_events.csv')
    
    testing_events = pd.read_csv(filepath + 'events.csv')
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
    
    #creating aggregated events
    aggregated_events = ''
    
    joined = pd.merge(testing_events, feature_map, how='left', on='event_id')
    filtered_events_not_null = joined.dropna()
       
    aggr = filtered_events_not_null.loc[filtered_events_not_null['event_id'].str.contains('D')]
    aggr = aggr[["patient_id","idx","value"]]
    aggr = aggr.groupby(["patient_id","idx"]).value.sum()
    aggr = aggr.reset_index()
       
    counted = filtered_events_not_null.loc[filtered_events_not_null['event_id'].str.contains('L')]
    counted = counted[["patient_id","idx","value"]]
    counted = counted.groupby(["patient_id","idx"]).value.count()
    counted = counted.reset_index()
    
    join1 = pd.concat([aggr,counted],axis=0) 
    max_df = join1.groupby('idx').value.max()
    max_df = max_df.reset_index()
    max_df = max_df.rename(columns = {'value':'max_value'})
    aggregated_events = pd.merge(join1, max_df,how='left',on=['idx'])
    aggregated_events['final_value'] = aggregated_events['value'] / aggregated_events['max_value']
    
    aggregated_events = aggregated_events[["patient_id","idx","final_value"]]
    aggregated_events = aggregated_events.rename(columns = {'idx':'feature_id','final_value':'feature_value'})
    
    #creating features
    patient_features = {}
    mortality1 = {}
    
    aggregated_events['tuples'] = aggregated_events[['feature_id', 'feature_value']].apply(tuple, axis=1)
    grouped = aggregated_events.groupby(['patient_id'])['tuples'].apply(list)
    patient_features = grouped.to_dict()
    mortality1 = dict([(i,[a ]) for i, a in zip(mortality.patient_id, mortality.label)])

    #creating svmlight files
    deliverable1 = open('../data/test/features_svmlight.test', 'wb')
    deliverable2 = open('../deliverables/test_features.txt', 'wb')
    
    deliverable1.write("");
    
    for key, items in patient_features.items():
        if key in mortality1.keys():
            string = str(float(mortality1[key])) + " " + ' '.join(("%d:%f" % (fid, float(fvalue)) for fid, fvalue in sorted(items))) + " "
        else:
            string = "0 " + ' '.join(("%d:%f" % (fid, float(fvalue)) for fid, fvalue in sorted(items))) + " "
        string += '\n'
        deliverable1.write(string)

    for key in patient_features:
        string = str(int(key)) + ' '
        value = sorted(patient_features[key], key=lambda x: x[0])
        for tup in value:
            string += str(int(tup[0])) + ':' + str("{:.6f}".format(tup[1])) + ' '
        deliverable2.write(string)
        deliverable2.write('\n')
        
    deliverable1.close()
    deliverable2.close()
    
    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    X_test, Y_test = utils.get_data_from_svmlight("../data/test/features_svmlight.test")
    return X_train,Y_train,X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
#    clf1 = LogisticRegression(random_state=1)
#    clf2 = RandomForestClassifier(random_state=1)
#    clf3 = GaussianNB()
#    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
#    eclf1 = eclf1.fit(X_train, Y_train)
#    answer1 = eclf1.predict(X_test)
    #lr = AdaBoostClassifier(n_estimators=40, random_state=545510477, learning_rate = 0.75) #better till date
#    kf = KFold(n_splits=5,random_state = 545510477)
#    
#    for train_index, test_index in kf.split(X_train, Y_train):
#        y_train1 = []
#        y_test1 = []
#        X_train1, X_test1 = X_train[train_index], X_train[test_index]
#        for i in train_index:
#            y_train1.append(Y_train[i])
#        for j in test_index:
#            y_test1.append(Y_train[j])
        
        #answer1 = models_partc.logistic_regression_pred(X_train1, y_train1, X_test1)
        
    lr = GradientBoostingClassifier(n_estimators=5000, random_state=545510477)
    lr.fit(X_train.toarray(),Y_train)
    answer1 = lr.predict(X_test.toarray())
    #print scores.mean()
    return answer1

def main():
    X_train, Y_train, X_test = my_features()
    Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
#    a1 = accuracy_score(Y_train, Y_pred)
#    auc1 = roc_auc_score(Y_train, Y_pred)
#    print "Accuracy: "+str(a1)
#    print "AUC: "+str(auc1)
	
    #print Y_pred
    utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	