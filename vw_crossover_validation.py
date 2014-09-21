from sklearn import cross_validation
import copy
import numpy as np
import math
from os import system
from sklearn import metrics




#   param_grid = {'--loss_function': ['logistic','hinge'], '--l2': [0.1,0,2]}
final_vals = {}


def roc_measure(true_labels,predictions):
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)    
    auc = metrics.auc(fpr,tpr)            
    return auc

def set_train_data(vw_train_loc,train_index,test_index):
    train_file = open("vw_cv_train.vw","wb")
    test_file = open("vw_cv_test.vw","wb")

    for r,line in enumerate(open(vw_train_loc)):
        if r in train_index:
            train_file.write(line)
        elif r in test_index:
            test_file.write(line)

    train_file.close()
    test_file.close()
        

def tuned_parameter_to_vw_command(tuned_parameter):
    recognized_list = ["--loss_function","--l2","--l1","-l","--initial_t","--power_t"]
    final_string = "vw -d vw_cv_train.vw -f model.vw "
    for ele in recognized_list:
        if ele in tuned_parameter:
            final_string = final_string + ele + " " + str(tuned_parameter[ele]) + " " 

    return final_string

def get_all_labels(vw_train_loc):
    labels = []
    for line in open(vw_train_loc):
        if(line[0]=="1"):
            labels.append(1)
        else:
            labels.append(-1)

    return labels

def logistic(x):
    return 1/(1+math.exp(-x))

def load_predictions():
    system("vw vw_cv_test.vw -t -i model.vw -p pred.txt")
    predictions = []
    for line in open("pred.txt"):
        predictions.append(int(logistic(float(line))>=0.5))

    return predictions

def recursive_parser(vw_train_loc,measure,n_fold,parameter_map,tuned_parameter={}):
    
    key = parameter_map.keys()[0]
    get_all_labels(vw_train_loc)
    if(len(parameter_map)==1):
        vals = parameter_map.pop(key)
        for val in vals:
            tuned_parameter[key] = val
            target = np.asarray(get_all_labels(vw_train_loc))
            skf = cross_validation.StratifiedKFold(target, n_folds=n_fold)
            avg_measure = float(0)
            command = tuned_parameter_to_vw_command(tuned_parameter)
            for train_index, test_index in skf:
                set_train_data(vw_train_loc,train_index,test_index)
                system(command)
                true_labels = target[test_index]
                predictions = load_predictions()
                avg_measure += roc_measure(true_labels,predictions)

            avg_measure /= n_fold
            final_vals[avg_measure] = tuned_parameter
    else:
        vals = parameter_map.pop(key)
        for val in vals:
            tuned_parameter[key] = val
            recursive_parser(vw_train_loc,roc_measure,n_fold,copy.copy(parameter_map),copy.copy(tuned_parameter))


    





"""
    skf = cross_validation.StratifiedKFold(target, n_folds=n_fold)
    for train_index, test_index in skf:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = y[train_index], y[test_index]
        param_grid = [
"""




recursive_parser('train.vw',roc_measure,5,  {'--loss_function': ['hinge'],'--l1':[0],'-l':[0.5],'--power_t':[1]})
print final_vals.keys()
print final_vals[max(final_vals.keys())]
#print load_predictions()
