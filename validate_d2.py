"""
Validate the selected biomarkers on independent dataset (called dataset2)
by constructing its own prediction model
"""

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn import cross_validation as cv
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, make_scorer, accuracy_score,f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix

def combine_row(l1, l2):
    l1 = pd.DataFrame(l1)
    l2 = pd.DataFrame(l2)
    l = l1.append(l2, ignore_index = True)
    return(l)

# split the data into training and testing while keeping the ratio of two categories
def train_test_equal_split(X_1, X_2, y_1, y_2, test_ratio, rand_seed_number):
    X_train1, X_test1, y_train1, y_test1 = cv.train_test_split(X_1, y_1, test_size=test_ratio, random_state=rand_seed_number)
    X_train2, X_test2, y_train2, y_test2 = cv.train_test_split(X_2, y_2, test_size=test_ratio, random_state=rand_seed_number)

    X_train = combine_row(X_train1, X_train2)
    y_train = combine_row(y_train1,y_train2)
    X_test = combine_row(X_test1, X_test2)
    y_test = combine_row(y_test1, y_test2)
    
    return(X_train, X_test, y_train, y_test)

## 1.0 read gene expression matrix
raw_metadata = pd.DataFrame.from_csv('GSE13105/GSE13015.tsv', sep='\t')
m1 = pd.DataFrame.from_csv('GSE13105/GPL6947_gene.txt', sep='\t')
m2 = pd.DataFrame.from_csv('GSE13105/GPL6106_gene.txt', sep='\t')
m12 = pd.concat([m1, m2], axis = 1).dropna(how='any')
raw_combined_matrix = pd.concat([raw_metadata, m12.T], axis = 1)

## 2.0 Label the categories of sepsis patients. Then focus on survivors and non-survivors only.
combined_matrix = raw_combined_matrix.copy()
combined_matrix['class'] = 0
condition = (combined_matrix['Illness'] == 'Sepsis/Melioidosis') & (combined_matrix['Outcome'] == 'Survivor')
combined_matrix.loc[condition, 'class'] = 1
condition = (combined_matrix['Illness'] == 'Sepsis/Melioidosis') & (combined_matrix['Outcome'] == 'Non survivor')
combined_matrix.loc[condition, 'class'] = 2
condition = combined_matrix['Illness'] == 'Control/healthy'
combined_matrix.loc[condition, 'class'] = 3
condition = combined_matrix['Illness'] == 'Control/Recovery'
combined_matrix.loc[condition, 'class'] = 4
condition = combined_matrix['Illness'] == 'Control/type 2 diabetes'
combined_matrix.loc[condition, 'class'] = 5
condition = (combined_matrix['Illness'] == 'Sepsis/Other infections') & (combined_matrix['Outcome'] == 'Survivor')
combined_matrix.loc[condition, 'class'] = 1
condition = (combined_matrix['Illness'] == 'Sepsis/Other infections') & ((combined_matrix['Outcome'] == 'Non survivor') | (combined_matrix['Outcome'] == 'Non survivor  '))
combined_matrix.loc[condition, 'class'] = 2

label = combined_matrix['class']
X_class_1 = combined_matrix[label == 1].iloc[:,:-1]
X_class_2 = combined_matrix[label == 2].iloc[:,:-1]
y_class_1 = combined_matrix[label == 1].iloc[:,-1]
y_class_2 = combined_matrix[label == 2].iloc[:,-1]

## 3.0 Test the biomarkers found from dataset1
gene_list_in_common_d1 = pd.Series.from_csv('d1_final_gene_list_in_common_500.csv')
gene_list_in_common = []
for key in gene_list_in_common_d1:
    if key in combined_matrix.columns[46:]: 
        print key
        gene_list_in_common.append(key)

## cross validation in the training dataset firstly
X_alltr.columns = X_class_1.columns
X_val.columns = X_class_1.columns

X_cv_train = X_alltr.loc[:,gene_list_in_common]
X_sep_test = X_val.loc[:, gene_list_in_common]
y_cv_train = y_alltr.copy()
y_cv_train[y_cv_train == -1] = 2
y_cv_train = y_cv_train.values.flatten()
y_sep_test = y_val.values.flatten()

X_train = X_cv_train
X_test = X_sep_test
y_train = y_cv_train
y_test = y_sep_test

# Using GridSearchCV to find the best values for C and gamma
C_range = 10.0 ** np.arange(-4, 4)
gamma_range = 10.0 ** np.arange(-20, 1)
param_grid = dict(gamma=gamma_range, C=C_range)
skf = cv.StratifiedKFold(y=y_train, n_folds=3)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=skf)
grid.fit(X_train, y_train)

# Print out parameters
crossclf = svm.SVC(probability=True, **grid.best_params_)
y_pred = crossclf.fit(X_train, y_train).predict(X_test)
print crossclf
print "Best parameter", grid.best_params_ # {'C': 10.0, 'gamma': 0.001}
print "Cross-Validation score", cv.cross_val_score(crossclf, X_train,y_train, cv=5).mean()
print "Independent test score", accuracy_score(y_test, y_pred)
print "Independent precision score", precision_score(y_test, y_pred)
print "Independent recall score", recall_score(y_test, y_pred)
print "Independent f1 score", f1_score(y_test, y_pred)

# Compute roc and auc
probas_ = crossclf.predict_proba(X_test)
y_test[y_test == 2] = 0
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 0])
roc_auc = auc(fpr, tpr)
print "Area under the curve", roc_auc  # 0.99

# Confusion Matrix
y_pred = crossclf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print "The confusion matrix:"
print cm

# Plot roc curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.show()
