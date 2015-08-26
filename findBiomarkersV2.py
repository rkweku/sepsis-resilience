"""
Read out the microarray gene expression profile and metadata.
Devide the data into training and validation parts,
then use 5-fold cross validation to select feature and test on validation data.
Output the biomarkers found.
Then use these gene expressions as feature to train model in cross validation
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.feature_selection import f_classif
from sklearn import cross_validation as cv
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, make_scorer,\
     accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Initialize empty dictionary for the groups to be compared
groupOneDict = {}
groupTwoDict = {}
#############################Execution Variables###############################
# The name of the file holding the expression data
expressionDataFilename = 'GSE66890/GSE66890_series_matrix.txt'

# The name of the file holding the metadata
metadataFilename = 'GSE66890/GSE66890_Metadata.tsv'

# Input the column name from the metadata file that has the required
# information to discern the various groups from each other
groupColumnName = 'Title'

# Only change the key value. Write the word or words that classifies a specific
# group. If there are multiple entries that fit into one group, add a new line
# and only replace the key word. **Key is case sensitive**
groupOneDict['sepsis'] = 1
#groupOneDict['Septic'] = 1

groupTwoDict['excluded'] = 2

# The name of the csv file to output the common gene list to. Specify directory
# and file extension.
outputFilename = 'GSE66890/commonGenes500.csv'
###############################################################################
def combine_row(l1, l2):
    """
    Combine rows for two dataframes

    Input:
        l1: First dataframe
        l2: Second dataframe

    Return:
        Combined l1 and l2
    """

    l1 = pd.DataFrame(l1)
    l2 = pd.DataFrame(l2)
    l = l1.append(l2, ignore_index = True)
    return(l)

def train_test_equal_split(X_1, X_2, y_1, y_2, test_ratio):
    """
    Split the test and trianing data based on the test_ratio.

    Input:
        X_1: dataframe with the data for group 1 for all samples including
             metadata and expression counts at each gene
        X_2: dataframe with the group ID of each sample for group 1
        Y_1: dataframe with the data for group 2 for all samples including
             metadata and expression counts at each gene
        Y_2: dataframe with the group ID of each sample for group 2
        test_ratio: Proportion of dataset that should be split for testing

    Return:
        Separate dataframes for each the training and test data as well as their
        group IDs

    """

    # Split the training data 
    X_train1, X_test1, y_train1, y_test1 = cv.train_test_split(X_1, y_1, test_size=test_ratio)
    X_train2, X_test2, y_train2, y_test2 = cv.train_test_split(X_2, y_2, test_size=test_ratio)

    X_train = combine_row(X_train1, X_train2)
    y_train = combine_row(y_train1,y_train2)
    X_test = combine_row(X_test1, X_test2)
    y_test = combine_row(y_test1, y_test2)
    
    return(X_train, X_test, y_train, y_test)

## 1.0 read gene expression matrix and metadata, then combine them
print("Step 1")
header = []
rawData = []

# Read in the full series file as a csv file
with open(expressionDataFilename, 'r') as f:
    # Set start flag to 0. Only want to read in data table
    startFlag = 0

    # Read file as wholeFile
    wholeFile = csv.reader(f, delimiter='\t')

    # Loop through each row of wholeFile
    for row in wholeFile:
        # If empty row, pass
        if(not row):
            pass
        # If the line ends with 'table_begin', set a flag to 1.
        # Need to save the next line as header and then the remainder
        # text will be data
        elif(row[0].endswith('table_begin')):
            startFlag = 1
        
        # If the line is 'table_end', this should be the end of the file
        # Set startFlag back to 0 to let loop end
        elif(row[0].endswith('table_end')):
            startFlag = 0

        # If we have found appropriate start, this line will the be header
        # So save this and set flag to 2 in preparation of reading the data
        elif(startFlag == 1):
            header = row
            startFlag = 2

        # While the startFlag is 2, save each line as rawData as this
        # should all be data to be saved
        elif(startFlag == 2):
            rawData.append(row)

# Create dataframe from the rawData and set columns to header
raw_matrix = pd.DataFrame(rawData, columns=header)

# Remove all columns that don't contain the expression counts or
# the name of the genes
idx = [('GSM' in x or x=='ID_REF') for x in raw_matrix.columns]
raw_matrix = raw_matrix.ix[:,idx]

# csv.reader creates string from all data. Need to change expression counts
# back to floats, but need to replace all instances of 'null' or empty string
# with np.nan because typecasting as float results in errors
raw_matrix = raw_matrix.applymap(lambda x: np.nan if x == 'null' else x) 

raw_matrix = raw_matrix.applymap(lambda  x: np.nan if x == '' else x)

# Change all expression counts to floats
sampleColumns = [col for col in raw_matrix.columns if 'GSM' in col]
raw_matrix[sampleColumns] = raw_matrix[sampleColumns].astype('float')

# Drop any row that contains null for an expression count
no_null_matrix = raw_matrix.dropna(how='any').T

# Transpose the no_null_matrix and set the idnex to the gene name
no_null_matrix = no_null_matrix.T.set_index('ID_REF')
#no_null_matrix.to_csv('test.txt')
# Read in the metadata file
raw_metadata = pd.DataFrame.from_csv(metadataFilename, sep='\t')

# Combine the metadata with the expression data
raw_combined_matrix = pd.concat([raw_metadata, no_null_matrix.T], axis = 1)

## 2.0 Label the three categories of sepsis patients. Then focus on survivors and non-survivors only.
print("Step 2")
all_matrix = raw_combined_matrix.copy()

for key in groupOneDict:
    condition = all_matrix[groupColumnName] .str.contains(key)
    all_matrix.loc[condition, 'class'] = groupOneDict[key]

for key in groupTwoDict:
    condition = all_matrix[groupColumnName] .str.contains(key)
    all_matrix.loc[condition, 'class'] = groupTwoDict[key]

allday_matrix= all_matrix[(all_matrix['class'] == 1) | (all_matrix['class'] == 2)]# | (combined_matrix['class'] == 3)]

label = allday_matrix['class']
X_class_1 = allday_matrix[label == 1].iloc[:,:-1]
X_class_2 = allday_matrix[label == 2].iloc[:,:-1]
y_class_1 = allday_matrix[label == 1].iloc[:,-1]
y_class_2 = allday_matrix[label == 2].iloc[:,-1]

## 3.0 Split the whole data into training and validation
print("Step 3")
X_alltr, X_val, y_alltr, y_val = train_test_equal_split(X_class_1, X_class_2, y_class_1, y_class_2, 0.33)
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
    X_tr, X_te, y_tr, y_te = train_test_equal_split(X_alltr_1, X_alltr_2, y_alltr_1, y_alltr_2, 0)
    X_tr.columns = X_class_1.columns
    X_tr_f = X_tr.iloc[:,33:]
    y_tr_f = y_tr.iloc[:,:].values.flatten()

    ## ANOVA
    F, pv = f_classif(X_tr_f, y_tr_f)
    ## Bonferroni correction for multiple tests
    pv = pv * int((y_tr == 1).sum()) * int((y_tr == 2).sum())
    X_tr_f_T = X_tr_f.T
    X_tr_f_T['F'] = pd.Series(F.astype(float),index=X_tr_f_T.index)
    X_tr_f_T['pv'] = pd.Series(pv.astype(float),index=X_tr_f_T.index)
    X_tr_f_T_sorted = X_tr_f_T.sort(columns = 'pv', ascending = 'False')
    pv_condition = X_tr_f_T_sorted['F'] < 0.001
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
    plt.legend([blue_dot, green_tri], ["Sepsis", "Controls"], loc=1)

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
pd.Series(gene_list_in_common).to_csv(outputFilename)

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
print 'y_pred: ', y_train
print 'y_pred: ', y_pred
print "Best parameter", grid.best_params_ # {'C': 10.0, 'gamma': 0.001}
print "Cross-Validation score", cv.cross_val_score(crossclf, X_train,y_train).mean()
print "Independent accuracy score", accuracy_score(y_test, y_pred)
print "Independent precision score", precision_score(y_test, y_pred)
print "Independent recall score", recall_score(y_test, y_pred)
print "Independent f1 score", f1_score(y_test, y_pred)

## 7.0 Plot ROC curve
# Compute roc and auc
print("Step 7")
probas_ = crossclf.predict_proba(X_test)
print probas_
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
