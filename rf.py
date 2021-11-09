"""
MIT License

Copyright (c) 2021 Yoga Suhas Kuruba Manjunath

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
from process_data import get_dataset, get_classes, plot
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys

import warnings
warnings.filterwarnings("ignore")

def eva_clf(y_true, y_pred_t):

    accuracy = accuracy_score(y_true, y_pred_t)
    precision = precision_score(y_true, y_pred_t, average='weighted',labels=np.unique(y_pred_t))
    recall = recall_score(y_true, y_pred_t,average='weighted')
    f1 = f1_score(y_true, y_pred_t, average='weighted',labels=np.unique(y_pred_t))

    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1 score: %f' % f1)
    
    print('classification report')
    print(classification_report(y_true, y_pred_t,labels=np.unique(y_pred_t)))
    
    print('confusion matrix')
    
    cf = confusion_matrix(y_true, y_pred_t)
    print((cf))
    
    FP = cf.sum(axis=0) - np.diag(cf)  
    FN = cf.sum(axis=1) - np.diag(cf)
    TP = np.diag(cf)
    TN = cf.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print("True Positive rate",TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print("True negative rate",TNR)
    
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print("positive predictive value",PPV)
    
    # Negative predictive value
    NPV = TN/(TN+FN)
    print("Negative predictive value",NPV) 
    
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print("false positive rate",FPR) 
    
    # False negative rate
    FNR = FN/(TP+FN)
    print("false negative rate",FNR) 
    
    # False discovery rate
    FDR = FP/(TP+FP)
    print("false discovery rate",FDR)     

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print("Accuracy",ACC)


    lbl = get_classes()    
    cf = confusion_matrix(y_true, y_pred_t, normalize='true')
    df_cm = pd.DataFrame(cf, index = [i for i in lbl["label"]],
                  columns = [i for i in lbl["label"]])
    plt.figure(figsize = (10,7))
    sn.set(font_scale=2) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})
    
    plt.tight_layout()
    plt.show()
    
   
    
def model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "./random_forest.joblib")
    y_pred_t = clf.predict(X_train)
    eva_clf(y_train, y_pred_t)    
    importances = clf.feature_importances_
    print(importances)     
    
def test(X_test, y_test):
    clf = joblib.load("./random_forest.joblib")
    y_pred_t = clf.predict(X_test)
    importances = clf.feature_importances_
    print(importances)  
    eva_clf( y_test, y_pred_t)    

def start(__data, __type):
    
    if __type == "train":
        X_train, y_train = get_dataset(__data, __type)
        model(X_train, y_train)
    elif __type == "test":
        X_test, y_test = get_dataset(__data, __type)    
        test(X_test, y_test)


def print_notice():
    print("Master please give me a correct command")
    print(" eg:") 
    print("     python rf.py dataset1")
    print("     python rf.py dataset2")
    print("     python rf.py plot dataset1")
    print("     python rf.py plot dataset2")

def main(argv):

    data_list=["dataset1", "dataset2"]

    if (len(argv) == 0 ):
        print_notice()
        sys.exit()      
    elif (len(argv) == 1 ):
        if ((argv[0] not in data_list)):
            print_notice()
            sys.exit()        
    elif (len(argv) == 2):
        if ( (argv[0] != "plot") and (argv[1] not in data_list) ):
            print_notice()
            sys.exit()            

    if argv[0] == "dataset1":
        start("dataset1", "train")
        start("dataset1", "test")
    elif argv[0] == "dataset2":
        start("dataset2", "train")
        start("dataset2", "test")      


    if argv[0] == "plot" and argv[1] == "dataset1":        
        plot("dataset1")
    elif argv[0] == "plot" and argv[1] == "dataset2":        
        plot("dataset2")        

if __name__ == "__main__":
    main(sys.argv[1:]) 
