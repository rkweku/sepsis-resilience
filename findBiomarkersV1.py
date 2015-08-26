"""
Read out the microarray gene expression profile and metadata.
Devide the data into training and validation parts,
then use 5-fold cross validation to select feature and test on validation data.
Output the biomarkers found.
Then use these gene expressions as feature to train model in cross validation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn import cross_validation as cv
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix

def combine_row(l1, l2):
    l1 = pd.DataFrame(l1)
    l2 = pd.DataFrame(l2)
    l = l1.append(l2, ignore_index = True)
    return(l)

def train_test_equal_split(X_1, X_2, y_1, y_2, test_ratio, rand_seed_number):
    X_train1, X_test1, y_train1, y_test1 = cv.train_test_split(X_1, y_1, test_size=test_ratio, random_state=rand_seed_number)
    X_train2, X_test2, y_train2, y_test2 = cv.train_test_split(X_2, y_2, test_size=test_ratio, random_state=rand_seed_number)

    X_train = combine_row(X_train1, X_train2)
    y_train = combine_row(y_train1,y_train2)
    X_test = combine_row(X_test1, X_test2)
    y_test = combine_row(y_test1, y_test2)
    
    return(X_train, X_test, y_train, y_test)

## 1.0 read gene expression matrix and metadata, then combine them
print("Step 1")
raw_matrix = pd.read_csv('GDS4971/GDS4971_full.soft', sep='\t', na_values=['null'])
idx = [('GSM' in x or x=='IDENTIFIER') for x in raw_matrix.columns]
raw_matrix = raw_matrix.ix[:,idx]
no_null_matrix = raw_matrix.dropna(how='any').T # d1

#idx = [(x=='ID_REF' or x=='IDENTIFIER') for x in raw_matrix.columns]
#raw_matrix_gene = raw_matrix.ix[:,idx]
#raw_matrix_gene = raw_matrix_gene.dropna(how='any')
#raw_matrix_gene = raw_matrix_gene.set_index('IDENTIFIER')
#raw_matrix_gene = raw_matrix_gene.iloc[:-1,:-1]
#print raw_matrix_gene
#exit()

no_null_matrix = no_null_matrix.T.set_index('IDENTIFIER')
raw_metadata = pd.DataFrame.from_csv('GDS4971/GDS4971.tsv', sep='\t')
raw_combined_matrix = pd.concat([raw_metadata, no_null_matrix.T], axis = 1)

## 2.0 Label the three categories of sepsis patients. Then focus on survivors and non-survivors only.
print("Step 2")
all_matrix = raw_combined_matrix.copy()
condition = all_matrix['group_day'].str.contains('^S_D')
all_matrix.loc[condition, 'class'] = 1
condition = all_matrix['group_day'].str.contains('^NS_D')
all_matrix.loc[condition, 'class'] = 2
condition = all_matrix['group_day'].str.contains('^HC_D')
all_matrix.loc[condition, 'class'] = 3
allday_matrix= all_matrix[(all_matrix['class'] == 1) | (all_matrix['class'] == 2)]# | (combined_matrix['class'] == 3)]

label = allday_matrix['class']
X_class_1 = allday_matrix[label == 1].iloc[:,:-1]
X_class_2 = allday_matrix[label == 2].iloc[:,:-1]
y_class_1 = allday_matrix[label == 1].iloc[:,-1]
y_class_2 = allday_matrix[label == 2].iloc[:,-1]

## 3.0 Split the whole data into training and validation
print("Step 3")
X_alltr, X_val, y_alltr, y_val = train_test_equal_split(X_class_1, X_class_2, y_class_1, y_class_2, 0.33, 15)
y_alltr.columns = ['class']
X_alltr_1 = X_alltr[y_alltr['class'] == 1]
X_alltr_2 = X_alltr[y_alltr['class'] == 2]
y_alltr_1 = y_alltr[y_alltr['class'] == 1]
y_alltr_2 = y_alltr[y_alltr['class'] == 2]

## 4.0 Among the training data, perform 5-fold cross validation.
## The concensus top 500 genes with p-value < 0.001 are the biomarkers.
print("Step 4")
gene_list = {}
gene_F = {}
# Set to 1 to just find the top 500 genes
cv_fold = 1
n_features_selected = 500
for i in range(1, cv_fold+1):
    print i
    X_tr, X_te, y_tr, y_te = train_test_equal_split(X_alltr_1, X_alltr_2, y_alltr_1, y_alltr_2, 0, i)
    X_tr.columns = X_class_1.columns
    X_tr_f = X_tr.iloc[:,32:]
    y_tr_f = y_tr.iloc[:,:].values.flatten()
    
    ## ANOVA
    F, pv = f_classif(X_tr_f, y_tr_f)
    ## Bonferroni correction for multiple tests
    pv = pv * int((y_tr == 1).sum()) * int((y_tr == 2).sum())
    X_tr_f_T = X_tr_f.T
    X_tr_f_T['F'] = pd.Series(F.astype(float),index=X_tr_f_T.index)
    X_tr_f_T['pv'] = pd.Series(pv.astype(float),index=X_tr_f_T.index)
    X_tr_f_T_sorted = X_tr_f_T.sort(columns = 'pv', ascending = 'False')
    pv_condition = X_tr_f_T_sorted['F'] < 0.005
    gene_list_tmp = pd.Series(X_tr_f_T_sorted[pv_condition].iloc[:n_features_selected,:].index)
    for j in gene_list_tmp:
        if j not in gene_list:
            gene_list[j] = 1
        else:
            gene_list[j] += 1
    
    count =1
    for j in gene_list_tmp:
        if count <10:
            print j + ':' + str(X_tr_f_T.loc[j, 'F'])
        count += 1
        if j not in gene_F:
            gene_F[j] = X_tr_f_T.loc[j, 'F']
        else:
            gene_F[j] += X_tr_f_T.loc[j, 'F']
   
    # plot the top2 feature
    y_tr.columns = ['class']
    label = y_tr['class']

    plt.clf()
    v1 = X_tr_f_T_sorted.T.iloc[:-2, 0]
    v2 = X_tr_f_T_sorted.T.iloc[:-2, 1]

    blue_dot, = plt.plot(v1[label == 1], v2[label == 1], 'bs')
    green_tri, = plt.plot(v1[label == 2], v2[label == 2], 'g^')
    plt.legend([blue_dot, green_tri], ["Survivors", "Non-survivors"], loc=1)

    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.show()

## 5.0 Find the consensus genes 
print("Step 5")
gene_list_in_common = []
for key in gene_list:
    if gene_list[key] == cv_fold:
        gene_list_in_common.append(key)  
print len(gene_list.keys())
pd.Series(gene_list_in_common).to_csv('d1_final_gene_list_in_common_' + str(n_features_selected) + '.csv')

## 6.0 Use expressions of these consensus genes as features to train model in cross validation
## cross validation in the training dataset firstly
print("Step 6")
X_alltr.columns = X_class_1.columns
X_val.columns = X_class_1.columns

X_cv_train = X_alltr.loc[:,gene_list_in_common]
X_sep_test = X_val.loc[:,gene_list_in_common]
y_cv_train = y_alltr.copy()

y_cv_train = y_cv_train.values.flatten()
y_sep_test = y_val.values.flatten()
y_cv_train[y_cv_train == 2] = 0
y_sep_test[y_sep_test == 2] = 0

X_train = X_cv_train
X_test = X_sep_test
y_train = y_cv_train
y_test = y_sep_test

# Using GridSearchCV to find the best values for C and gamma
C_range = 10.0 ** np.arange(-4, 4)
gamma_range = 10.0 ** np.arange(-10, 1)
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
print "Independent accuracy score", accuracy_score(y_test, y_pred)
print "Independent precision score", precision_score(y_test, y_pred)
print "Independent recall score", recall_score(y_test, y_pred)
print "Independent f1 score", f1_score(y_test, y_pred)

## 7.0 Plot ROC curve
# Compute roc and auc
print("Step 7")
probas_ = crossclf.predict_proba(X_test)
print probas_
print y_test
print y_pred
y_test[y_test == 2] = 0
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
print fpr, tpr
roc_auc = auc(fpr, tpr)
print "Area under the curve", roc_auc 

# Confusion Matrix
y_pred = crossclf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print "The confusion matrix:"
print cm

# Plot roc curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
