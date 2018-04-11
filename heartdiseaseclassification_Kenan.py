import numpy as np
import urllib
import matplotlib.pyplot as plt # Visuals
import seaborn as sns 
import sklearn as skl
import pandas as pd
from __future__ import division

from sklearn.cross_validation import train_test_split # Create training and test sets
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.tree import tree 
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import svm #SVM
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # AUC 
from sklearn.model_selection import KFold, cross_val_score #cross validation 
from sklearn import cross_validation  #cross validation 
import urllib2

import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as pt
plt.style.use('ggplot')
pt.set_credentials_file(username='ksakarc', api_key='vdo1Wqt73LV5vgWxTZyC')

Cleveland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Hungarian_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Switzerland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'
np.set_printoptions(threshold=np.nan) #see a whole array when we output it

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
ClevelandHeartDisease = pd.read_csv(urllib2.urlopen(Cleveland_data_URL), names = names) #gets Cleveland data
HungarianHeartDisease = pd.read_csv(urllib2.urlopen(Hungarian_data_URL), names = names) #gets Hungary data
SwitzerlandHeartDisease = pd.read_csv(urllib2.urlopen(Switzerland_data_URL), names = names) #gets Switzerland data
datatemp = [ClevelandHeartDisease, HungarianHeartDisease, SwitzerlandHeartDisease] #combines all arrays into a list

heartDisease = pd.concat(datatemp)#combines list into one array
heartDisease.head()

del heartDisease['ca']
del heartDisease['slope']
del heartDisease['thal']

heartDisease = heartDisease.replace('?', np.nan)
heartDisease.dtypes

def classImbalance(item):
    item_0 = 0
    item_1 = 0
    item_2 = 0
    item_3 = 0
    item_4 = 0
    item_5 = 0
    for item_i in heartDisease[item]:
        for i in range(0,6):
            if (item == i):
                item_i +=1
    heartDisease_i = 0
    for i in  range (0,6):
        heartDisease_i = (item_i/len(heartDisease)) * 100
        print("The percentage of level", i, "in the response variable is: {0:.2f}".format(heartDisease_i)) 
 
classImbalance('heartdisease')



#Create a boxplot of data

trace0 = go.Box(
    y=heartDisease['age'],
    name='age'
)
trace1 = go.Box(
    y=heartDisease['sex'],
    name='sex'
)
trace2 = go.Box(
    y=heartDisease['cp'],
    name='cp'
)
trace3 = go.Box(
    y=heartDisease['trestbps'],
    name='trestbps'
)
trace4 = go.Box(
    y=heartDisease['chol'],
    name='chol'
)
trace5 = go.Box(
    y=heartDisease['fbs'],
    name='fbs'
)
trace6 = go.Box(
    y=heartDisease['restecg'],
    name='restecg'
)
trace7 = go.Box(
    y=heartDisease['thalach'],
    name='thalach'
)
trace8 = go.Box(
    y=heartDisease['exang'],
    name='exang'
)
trace9 = go.Box(
    y=heartDisease['oldpeak'],
    name='oldpeak'
)
trace10 = go.Box(
    y=heartDisease['heartdisease'],
    name='heart disease status'
)
plotdata = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]
py.iplot(plotdata)


#we need to normalize the data

for item in heartDisease: #converts everything to floats
    heartDisease[item] = pd.to_numeric(heartDisease[item])

def normalize(heartDisease, toNormalize): #normalizes 
    result = heartDisease.copy()
    for item in heartDisease.columns:
        if (item in toNormalize):
            max_value = heartDisease[item].max()
            min_value = heartDisease[item].min()
            result[item] = (heartDisease[item] - min_value) / (max_value - min_value)
    return result
toNormalize = ['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak'] #columns to normalize
heartDisease = normalize(heartDisease, toNormalize)
heartDisease = heartDisease.dropna()
heartDisease.head()

for i in range(1,5):
    heartDisease['heartdisease'] = heartDisease['heartdisease'].replace(i,1)
    
f, ax = plt.subplots(figsize=(11, 15))

ax.set_axis_bgcolor('#fafafa')
plt.title("Box Plot of Transformed Data Set")
ax.set(xlim=(-.05, 1.05))
ax = sns.boxplot(data = heartDisease[1:29], orient = 'h', palette = 'Set2')
plt.show()

#We are creating the test and training splits now

train, test = train_test_split(heartDisease, test_size = 0.20, random_state = 42)
# Create the training test omitting the diagnosis

training_set = train.iloc[:, train.columns != 'heartdisease']
# Next we create the class set 
class_set = train.iloc[:, train.columns == 'heartdisease']

# Next we create the test set doing the same process as the training set
test_set = test.iloc[:, test.columns != 'heartdisease']
test_class_set = test.loc[:, test.columns == 'heartdisease']

dt = tree.DecisionTreeClassifier()
dt = dt.fit(train[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']], train['heartdisease'])
predictions_dt = dt.predict(test[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']])
predictright = 0
predictions_dt.shape[0]
for i in range(0,predictions_dt.shape[0]-1):
    if (predictions_dt[i]== test.iloc[i][10]):
        predictright +=1
accuracy = predictright/predictions_dt.shape[0]
accuracy

#ROC curve calculation 
fpr2, tpr2, _ = roc_curve(predictions_dt, test_class_set)

#AUC curve calcuation
auc_dt = auc(fpr2, tpr2)


fitRF = RandomForestClassifier(random_state = 42, 
                                criterion='gini',
                                n_estimators = 500,
                                max_features = 5)

fitRF.fit(training_set, class_set['heartdisease'])

importancesRF = fitRF.feature_importances_
indicesRF = np.argsort(importancesRF)[::-1]
indicesRF

namesInd = names[:11]
print("Feature ranking:")

for f in range(10):
    i = f
    print("%d. The feature '%s' has a Gini Importance of %f" % (f + 1, 
                                                                namesInd[indicesRF[i]], 
                                                                importancesRF[indicesRF[f]]))
    
feature_space = []
for i in range(9, -1, -1):
    feature_space.append(namesInd[indicesRF[i]])
    
indRf = sorted(importancesRF) #sort into decreasing order 
index = np.arange(10) #arrange the importance

f, ax = plt.subplots(figsize=(11, 11))

ax.set_axis_bgcolor('#fafafa')
plt.title('Feature importances for Random Forest Model')
plt.barh(index, indRf,
    	align="center", 
    	color = '#875FDB')
plt.yticks(index, ('cp', 'exang', 'oldpeak', 'chol', 'thalach', 'age', 'trestbps', 'sex', 'restecg', 'fbs'))
plt.ylim(-1, 10)
plt.xlim(0, 0.15)
plt.xlabel('Gini Importance')
plt.ylabel('Feature')

plt.show()

predictions_RF = fitRF.predict(test_set)
print(pd.crosstab(predictions_RF, test_class_set['heartdisease'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))

accuracy_RF = fitRF.score(test_set, test_class_set['heartdisease'])

print("Here is our mean accuracy on the test set:\n",
     '%.3f' % (accuracy_RF * 100), '%')

# Here we calculate the test error rate!
test_error_rate_RF = 1 - accuracy_RF
print("The test error rate for our model is:\n",
     '%.3f' % (test_error_rate_RF * 100), '%')

#ROC curve calculation
fpr1, tpr1, _ = roc_curve(predictions_RF, test_class_set)
#AUC curve calcuation
auc_rf = auc(fpr1, tpr1)

#KNN

heartDiseaseKnn = KNeighborsClassifier(n_neighbors=10)
heartDiseaseKnn.fit(training_set, class_set['heartdisease'])


# We predict the class for our training set
predictionsTrain = heartDiseaseKnn.predict(training_set) 

# Here we create a matrix comparing the actual values vs. the predicted values
print(pd.crosstab(predictionsTrain, class_set['heartdisease'], 
                  rownames=['Predicted Values'], colnames=['Actual Values']))

# Measure the accuracy based on the trianing set
accuracyTrain = heartDiseaseKnn.score(training_set, class_set['heartdisease'])

print("Here is our accuracy for our training set:")
print('%.3f' % (accuracyTrain * 100), '%')


print("The train error rate for our model is:")
train_error_rate = 1 - accuracyTrain  
print('%.3f' % (train_error_rate * 100), '%')

# First we predict the Dx for the test set and call it predictions
predictions = heartDiseaseKnn.predict(test_set)

# Let's compare the predictions vs. the actual values
print(pd.crosstab(predictions, test_class_set['heartdisease'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))

# Let's get the accuracy of our test set
accuracy = heartDiseaseKnn.score(test_set, test_class_set['heartdisease'])

# TEST ERROR RATE!!
print("Here is our accuracy for our test set:")
print('%.3f' % (accuracy * 100), '%')

n = KFold(n_splits=3)

scores = cross_val_score(heartDiseaseKnn, 
                         test_set, 
                         test_class_set['heartdisease'], cv = n)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

fpr3, tpr3, _ = roc_curve(predictions, test_class_set)
auc_knn = auc(fpr3, tpr3)

fig, ax = plt.subplots(figsize=(10, 10))

plt.plot(fpr1, tpr1,label='Decision Trees ROC Curve (area = %.4f)' % auc_dt, 
         color = 'navy', 
         linewidth=2)
plt.plot(fpr2, tpr2,label='Random Forest ROC Curve (area = %.4f)' % auc_rf, 
         color = 'red', 
         linestyle=':', 
         linewidth=2)
plt.plot(fpr3, tpr3,label='Kth Nearest Neighbor ROC Curve (area = %.4f)' % auc_knn, 
         color = 'red', 
         linestyle=':', 
         linewidth=2)


ax.set_axis_bgcolor('#fafafa')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison For All Models')
plt.legend(loc="lower right")

plt.show()

trace0 = go.Box(
    y=heartDisease['age'],
    name='age'
)
trace1 = go.Box(
    y=heartDisease['sex'],
    name='sex'
)
trace2 = go.Box(
    y=heartDisease['cp'],
    name='cp'
)
trace3 = go.Box(
    y=heartDisease['trestbps'],
    name='trestbps'
)
trace4 = go.Box(
    y=heartDisease['chol'],
    name='chol'
)
trace5 = go.Box(
    y=heartDisease['fbs'],
    name='fbs'
)
trace6 = go.Box(
    y=heartDisease['restecg'],
    name='restecg'
)
trace7 = go.Box(
    y=heartDisease['thalach'],
    name='thalach'
)
trace8 = go.Box(
    y=heartDisease['exang'],
    name='exang'
)
trace9 = go.Box(
    y=heartDisease['oldpeak'],
    name='oldpeak'
)
trace10 = go.Box(
    y=heartDisease['heartdisease'],
    name='heart disease status'
)
plotdata = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]
py.iplot(plotdata)

heartDisease.describe()