import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean
from sklearn.metrics import accuracy_score, roc_auc_score

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    kf = KFold(n_splits=k,random_state = 545510477)
    sum_acc = 0
    sum_auc = 0
    
    for train_index, test_index in kf.split(X, Y):
        y_train1 = []
        y_test1 = []
        X_train1, X_test1 = X[train_index], X[test_index]
        for i in train_index:
            y_train1.append(Y[i])
        for j in test_index:
            y_test1.append(Y[j])
        
        answer1 = models_partc.logistic_regression_pred(X_train1, y_train1, X_test1)
        a1 = accuracy_score(y_test1, answer1)
        auc1 = roc_auc_score(y_test1, answer1)
        
        sum_acc += a1
        sum_auc += auc1
    
        
    return sum_acc/k,sum_auc/k


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    
    ss = ShuffleSplit(test_size = test_percent, n_splits=iterNo, random_state = 545510477)
    sum_acc = 0
    sum_auc = 0
    
    for train_index, test_index in ss.split(X, Y):
        y_train1 = []
        y_test1 = []
        X_train1, X_test1 = X[train_index], X[test_index]
        for i in train_index:
            y_train1.append(Y[i])
        for j in test_index:
            y_test1.append(Y[j])
        
        answer1 = models_partc.logistic_regression_pred(X_train1, y_train1, X_test1)
        a1 = accuracy_score(y_test1, answer1)
        auc1 = roc_auc_score(y_test1, answer1)
        
        sum_acc += a1
        sum_auc += auc1
    
        
    return sum_acc/iterNo,sum_auc/iterNo


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()