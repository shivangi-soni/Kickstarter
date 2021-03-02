# -*- coding: utf-8 -*-
"""
INDIVIDUAL PROJECT - KICKSTARTER
@author: Shivangi Soni 
"""

#Please run the Developing the Model and Grading Sections (from line 19 to 309)
#Please input the training datset in line 24 and test dataset in line 222
#Appendix section has all the work that was done to come up with the models for regression, classification, and clustering 




####---------------------------------------------------------------Developing the model  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------------------------------


#import libraries
import pandas as pd
import numpy
import seaborn as sns
import matplotlib.pyplot as plt

###importing the dataset and pre-processing 
kickstarter_train_df = pd.read_excel(r"C:\Users\shiva\Downloads\Kickstarter.xlsx")
kickstarter_train_df = kickstarter_train_df[(kickstarter_train_df.state == 'successful')|(kickstarter_train_df.state == 'failed') ]
#train_df is used as the dataset for all the models 
train_df = kickstarter_train_df.drop(columns = ["launch_to_state_change_days", "backers_count", "project_id", "name","disable_communication","spotlight","currency", "staff_pick", "deadline","state_changed_at", "created_at", "launched_at","launched_at_yr","launched_at_month", "launched_at_hr","created_at_yr", "pledged", "name_len", "blurb_len","state_changed_at_hr", "state_changed_at_day", "state_changed_at_month", "state_changed_at_yr", "state_changed_at_weekday"])
train_df = train_df.dropna()
#Since days and hours have over 20 catgeories, creating bins for these variable  to avoid a lot of dummy variables 
hours_bins= [0,4,8,12,16,20,25]
days_bins = [0,7,14,21,32] 
train_df['deadline_day'] = pd.cut(train_df['deadline_day'], days_bins) 
train_df['deadline_hr'] = pd.cut(train_df['deadline_hr'], hours_bins) 
train_df['created_at_day'] = pd.cut(train_df['created_at_day'], days_bins) 
train_df['created_at_hr'] = pd.cut(train_df['created_at_hr'], hours_bins) 
train_df['launched_at_day'] = pd.cut(train_df['launched_at_day'], days_bins) 
train_df['goal'] = train_df['goal']* train_df['static_usd_rate']
train_df = train_df.drop(columns = ['static_usd_rate'])
train_df.reset_index(inplace = True, drop = True)
country = ['LU','SG','HK', 'BE', 'MX', 'AT', 'NO']
for i in range(len(train_df)):
    if train_df['country'][i] in country:
        train_df['country'] = train_df['country'].replace([train_df['country'][i]],'Others')

category = ['Comedy','Academic','Blues', 'Webseries', 'Thrillers', 'Shorts']
for i in range(len(train_df)):
    if train_df['category'][i] in category:
        train_df['category'] = train_df['category'].replace([train_df['category'][i]],'Others')        


#******************************************************REGRESSION MODEL****************************************************************************************************************************************************

### Setup the variables
X_reg = train_df[["goal","country","category", "name_len_clean", "blurb_len_clean","deadline_weekday","created_at_weekday","launched_at_weekday","deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day","create_to_launch_days", "launch_to_deadline_days"]]
y_reg = train_df["usd_pledged"]
#create dummy variables
X_reg = pd.get_dummies(X_reg, columns = ["country","category", "deadline_weekday",	"created_at_weekday","launched_at_weekday",	"deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day"])

### Split the data
from sklearn.model_selection import train_test_split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size = 0.33, random_state = 0)

#drop the outliers 
X_reg_train.reset_index(inplace = True, drop = True)
X_reg_test.reset_index(inplace = True,  drop = True)

y_reg_train = y_reg_train.to_frame()
y_reg_train.reset_index(inplace = True, drop = True)

y_reg_test = y_reg_test.to_frame()
y_reg_test.reset_index(inplace = True,  drop = True)

#getting rid of outliers 

from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators = 100,  contamination = 0.02, random_state =0) 
pred = iforest.fit_predict(X_reg_train) 
score = iforest.decision_function(X_reg_train)

# Extracting anomalies
from numpy import where
anom_index = where(pred== -1)
values = X_reg_train.iloc[anom_index]
X_reg_train= X_reg_train.drop(anom_index[0], axis = 0)
y_reg_train = y_reg_train.drop(anom_index[0], axis = 0)


#feature analysis results 

X_train_reg1 = X_reg_train.drop(columns = ['deadline_weekday_Friday', 'created_at_day_(21, 32]'])
X_test_reg1 =  X_reg_test.drop(columns =['deadline_weekday_Friday', 'created_at_day_(21, 32]'])


#Build the regression model
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error

random_forest = RandomForestRegressor(n_estimators = 100, random_state = 0, max_features = "sqrt")
model = random_forest.fit(X_train_reg1,y_reg_train.values.ravel())
# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(X_test_reg1)
# Calculate the mean squared error of the prediction
mse_train = mean_squared_error(y_reg_test, y_test_pred)
print("The MSE of training dataset", mse_train)


#******************************************************CLASSIFICATION MODEL****************************************************************************************************************************************************
### Setup the variables

X_class = train_df[["goal","country","category", "name_len_clean", "blurb_len_clean","deadline_weekday", "created_at_weekday","launched_at_weekday","deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day","create_to_launch_days", "launch_to_deadline_days"]]
y_class = train_df["state"]

#create dummy variables of categorical variables
X_class = pd.get_dummies(X_class, columns = ["country","category", "deadline_weekday",	"created_at_weekday","launched_at_weekday",	"deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day"])

### Split the data
from sklearn.model_selection import train_test_split
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size = 0.33, random_state = 0)

#drop the outliers 

X_class_train.reset_index(inplace = True, drop = True)
X_class_test.reset_index(inplace = True,  drop = True)

y_class_train = y_class_train.to_frame()
y_class_train.reset_index(inplace = True, drop = True)

y_class_test = y_class_test.to_frame()
y_class_test.reset_index(inplace = True,  drop = True)

from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators = 100,  contamination = 0.02, random_state =0) 
pred = iforest.fit_predict(X_class_train) 
score = iforest.decision_function(X_class_train)

# Extracting anomalies
from numpy import where
anom_index = where(pred== -1)
values = X_class_train.iloc[anom_index]

#for i in values_index:
X_class_train= X_class_train.drop(anom_index[0], axis = 0)
y_class_train = y_class_train.drop(anom_index[0], axis = 0)

#label encode the variables 
#y_class_train.values.reshape(len(y_class_train),)
#y_class_test.values.reshape(len(y_class_test),)

from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
y_class_train = lab_enc.fit_transform(y_class_train.values.flatten())
y_class_test = lab_enc.fit_transform(y_class_test.values.flatten())
#feature analysis
X_train_class = X_class_train.drop(columns = ['deadline_yr_2011','country_DE', 'category_Places', 'country_FR', 'category_Makerspaces', 'country_IT', 'category_Others', 'deadline_yr_2010','country_Others', 'country_NL', 'country_ES', 'country_IE', 'country_CH','country_NZ','country_DK' ,'country_SE', 'deadline_yr_2009', 'category_Robots', 'category_Immersive', 'country_AU', 'category_Flight', 'category_Spaces', 'deadline_yr_2017'])
X_test_class = X_class_test.drop(columns = ['deadline_yr_2011','country_DE', 'category_Places', 'country_FR', 'category_Makerspaces', 'country_IT', 'category_Others', 'deadline_yr_2010','country_Others', 'country_NL', 'country_ES', 'country_IE', 'country_CH','country_NZ','country_DK' ,'country_SE', 'deadline_yr_2009', 'category_Robots', 'category_Immersive', 'country_AU', 'category_Flight', 'category_Spaces', 'deadline_yr_2017'])

# Build the classification model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
gbt_class = GradientBoostingClassifier(random_state=0, n_estimators = 200)
model_gbt = gbt_class.fit(X_train_class, y_class_train)
# Using the model to predict the results based on the test dataset
y_test_pred_c1 = model_gbt.predict(X_test_class)
acc_gbt_class= accuracy_score(y_class_test,y_test_pred_c1) 
# Calculate the accuracy score of the prediction
print("Accuracy from training dataset:", acc_gbt_class)


#******************************************************CLUSTERING MODEL****************************************************************************************************************************************************
X_clust = kickstarter_train_df[["goal","name_len_clean","blurb_len_clean","static_usd_rate", "launch_to_deadline_days", "state", "backers_count", "staff_pick"]]
X_clust = X_clust.dropna()
X_clust['goal'] = X_clust['goal']*X_clust['static_usd_rate']
X_clust = X_clust.drop(columns = ["static_usd_rate"])

from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
X_clust["state"] = lab_enc.fit_transform(X_clust["state"])
X_clust["staff_pick"] = lab_enc.fit_transform(X_clust["staff_pick"])

#reset the index
X_clust.reset_index(inplace = True, drop = True)
#getting rid of outliers 
# Create isolation forest model

from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators = 100,  contamination = 0.05, random_state =0) 
pred = iforest.fit_predict(X_clust) 
score = iforest.decision_function(X_clust)
# Extracting anomalies
from numpy import where
anom_index = where(pred== -1)
values = X_clust.iloc[anom_index]
X_clust = X_clust.drop(anom_index[0], axis = 0)

#standardizing the data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std_clust = scaler.fit_transform(X_clust)

#K Means 
from sklearn.cluster import KMeans
#Optimal number of clusters found using Elbow Method (shown in Appendix)
kmeans1 = KMeans(n_clusters=7)
model_clust1 = kmeans1.fit(X_std_clust)
labels = model_clust1.predict(X_std_clust)

from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_std_clust,labels)
df = pd.DataFrame({'label':labels,'silhouette':silhouette})
print('Average Silhouette Score for Cluster 0: ',numpy.average(df[df['label'] == 0].silhouette))
print('Average Silhouette Score for Cluster 1: ',numpy.average(df[df['label'] == 1].silhouette))
print('Average Silhouette Score for Cluster 2: ',numpy.average(df[df['label'] == 2].silhouette))
print('Average Silhouette Score for Cluster 3: ',numpy.average(df[df['label'] == 3].silhouette))
print('Average Silhouette Score for Cluster 4 ',numpy.average(df[df['label'] == 4].silhouette))
print('Average Silhouette Score for Cluster 5: ',numpy.average(df[df['label'] == 5].silhouette))
print('Average Silhouette Score for Cluster 6: ',numpy.average(df[df['label'] == 6].silhouette))


####--------------------------------------------------------------------Grading  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------------------------------

###importing the dataset and pre-processing 
kickstarter_grading_df = pd.read_excel(r"C:\Users\shiva\Downloads\Kickstarter-Grading-Sample.xlsx")
kickstarter_grading_df = kickstarter_grading_df[(kickstarter_grading_df.state == 'successful')|(kickstarter_grading_df.state == 'failed') ]

#test_df is used as the dataset for all the models 
test_df = kickstarter_train_df.drop(columns = ["launch_to_state_change_days", "backers_count", "project_id", "name","disable_communication","spotlight","currency", "staff_pick", "deadline","state_changed_at", "created_at", "launched_at","launched_at_yr","launched_at_month", "launched_at_hr","created_at_yr", "pledged", "name_len", "blurb_len","state_changed_at_hr", "state_changed_at_day", "state_changed_at_month", "state_changed_at_yr", "state_changed_at_weekday"])
test_df = test_df.dropna()
#Since days and hours have over 20 catgeories, creating bins for these variable  to avoid a lot of dummy variables 
hours_bins= [0,4,8,12,16,20,25]
days_bins = [0,7,14,21,32] 
test_df['deadline_day'] = pd.cut(test_df['deadline_day'], days_bins) 
test_df['deadline_hr'] = pd.cut(test_df['deadline_hr'], hours_bins) 
test_df['created_at_day'] = pd.cut(test_df['created_at_day'], days_bins) 
test_df['created_at_hr'] = pd.cut(test_df['created_at_hr'], hours_bins) 
test_df['launched_at_day'] = pd.cut(test_df['launched_at_day'], days_bins) 
test_df['goal'] = test_df['goal']* test_df['static_usd_rate']
test_df = test_df.drop(columns = ['static_usd_rate'])
test_df.reset_index(inplace = True, drop = True)
country = ['LU','SG','HK', 'BE', 'MX', 'AT', 'NO']
for i in range(len(test_df)):
    if test_df['country'][i] in country:
        test_df['country'] = test_df['country'].replace([test_df['country'][i]],'Others')

category = ['Comedy','Academic','Blues', 'Webseries', 'Thrillers', 'Shorts']
for i in range(len(test_df)):
    if test_df['category'][i] in category:
        test_df['category'] = test_df['category'].replace([test_df['category'][i]],'Others') 

#********************************************************************REGRESSION MODEL********************************************************************************************************************************************************************************************************************

### Setup the variables
X_grading = test_df[["goal","country","category", "name_len_clean", "blurb_len_clean","deadline_weekday","created_at_weekday","launched_at_weekday","deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day","create_to_launch_days", "launch_to_deadline_days"]]
y_grading = test_df["usd_pledged"]

#create dummy variables
X_grading = pd.get_dummies(X_grading, columns = ["country","category", "deadline_weekday",	"created_at_weekday","launched_at_weekday",	"deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day"])


# Get missing columns in the training test so that if there is any dummy category missing, that can be taken care of 
missing_columns = set(X_train_reg1.columns) - set(X_grading.columns)
# Set default value of 0 for any missing column 
for i in missing_columns:
    X_grading[i] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_grading = X_grading[X_train_reg1.columns]

#Build the regression model
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error

# Using the model to predict 
y_grading_pred = model.predict(X_grading)
# Calculate the mean squared error of the prediction
mse_grading = mean_squared_error(y_grading, y_grading_pred)
print("The MSE of grading dataset", mse_grading)

#***********************************************************************CLASSIFICATION MODEL*********************************************************************************************************************************************************************************************************************************************************

X_class_grading = test_df[["goal","country","category", "name_len_clean", "blurb_len_clean","deadline_weekday", "created_at_weekday","launched_at_weekday","deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day","create_to_launch_days", "launch_to_deadline_days"]]
y_class_grading = test_df["state"]

#create dummy variables of categorical variables
X_class_grading = pd.get_dummies(X_class_grading, columns = ["country","category", "deadline_weekday",	"created_at_weekday","launched_at_weekday",	"deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day"])

from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
y_class_grading = lab_enc.fit_transform(y_class_grading.values.flatten())

# Get missing columns in the training test so that if there is any dummy category missing, that can be taken care of 
missing_cols = set(X_train_class.columns) - set(X_class_grading.columns)
# Set default value of 0 for any missing column 
for i in missing_cols:
    X_class_grading[i] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_class_grading = X_class_grading[X_train_class.columns]

# Build the classification model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Using the model to predict the state
y_grading_pred_class= model_gbt.predict(X_class_grading)
acc_gbt_class_grading= accuracy_score(y_class_grading, y_grading_pred_class) 
# Calculate the accuracy score of the prediction
print("Accuracy from grading dataset:", acc_gbt_class_grading)

print("Results:")
print("The MSE of grading dataset", mse_grading)
print("Accuracy from grading dataset:", acc_gbt_class_grading)




####-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
#                                           APPENDIX 
#
####------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
kickstarter_train_df = pd.read_excel(r"C:\Users\shiva\Downloads\Kickstarter.xlsx")


###--------------------------------- Data Pre-Processing ----------------------------------------------------------------------------------
#only include observations with successful and failed projects 
kickstarter_train_df = kickstarter_train_df[(kickstarter_train_df.state == 'successful')|(kickstarter_train_df.state == 'failed') ]

#Exploratory Analysis 
#Checking proprotion of categroical variables 
kickstarter_train_df.disable_communication.value_counts(normalize=True)
kickstarter_train_df.staff_pick.value_counts(normalize=True)

#checking how many null values 
kickstarter_train_df.info()


#Creating train_df which will have all the importatnt features based on exploratory analysisand will be used for further analysis 
#launch_to_state_change_days is the one with most null values (5386 null values); hence, drop the column 
#project_id and name don't impact the target variable
#disable_communication is dropped since around all of values are false
#preliminary analysis showed that the deadline time and start_changed_at time are identical so all paramters associated to state_changed_at were dropped
#since all paramters including deadline, launched_at, and created_at are split into hr, day, month, year, columns, these were dropped too  
#currency and country are highly correlated
#spotlight,staff_pick and bakcer's count are invalid features because they are realized after 
#Didn't drop usd_pledged and state since they will be target variables later
train_df = kickstarter_train_df.drop(columns = ["launch_to_state_change_days", "backers_count", "project_id", "name","disable_communication","spotlight","currency", "staff_pick", "deadline","state_changed_at", "created_at", "launched_at","launched_at_yr","launched_at_month", "launched_at_hr","created_at_yr", "pledged", "name_len", "blurb_len","state_changed_at_hr", "state_changed_at_day", "state_changed_at_month", "state_changed_at_yr", "state_changed_at_weekday"])
train_df.shape

#drop all observations with null values (obs after dropping = 14214)
train_df = train_df.dropna()
#Since days and hours have over 20 catgeories, creating bins for these variable  to avoid a lot of dummy variables 
hours_bins= [0,4,8,12,16,20,25]
days_bins = [0,7,14,21,32] 

train_df['deadline_day'] = pd.cut(train_df['deadline_day'], days_bins) 
train_df['deadline_hr'] = pd.cut(train_df['deadline_hr'], hours_bins) 
train_df['created_at_day'] = pd.cut(train_df['created_at_day'], days_bins) 
train_df['created_at_hr'] = pd.cut(train_df['created_at_hr'], hours_bins) 
train_df['launched_at_day'] = pd.cut(train_df['launched_at_day'], days_bins) 
#train_df['launched_at_hr'] = pd.cut(train_df['launched_at_hr'], hours_bins) #removed later 

#Since usd_pledged is in USD, goal should be in USD as well for comparison purposes
train_df['goal'] = train_df['goal']* train_df['static_usd_rate']
#can remove statitc_usd_rate as well now 
train_df = train_df.drop(columns = ['static_usd_rate'])


#checking how many unique countries and how many projects for each country 
num_countries = train_df.country.nunique()
countries = train_df.groupby('country')['goal'].count()
#any country which has lower than 50 projects is classified as others 
train_df['country'] = train_df['country'].replace(['LU','SG','HK', 'BE', 'MX', 'AT', 'NO'],'Others')

#checking how many unique categories and how many projects for each category
train_df.category.nunique()
categories= train_df.groupby('category')['goal'].count()
#any category which has lower than 50 projects is classified as others 
train_df['category'] = train_df['category'].replace(['Comedy','Academic','Blues', 'Webseries', 'Thrillers', 'Shorts'],'Others')

###Regression Model 
#define x and y variables
#X has all the avriables that train_df has except state and usd_pledged 
X_reg = train_df[["goal","country","category", "name_len_clean", "blurb_len_clean","deadline_weekday","created_at_weekday","launched_at_weekday",	"deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day","create_to_launch_days", "launch_to_deadline_days"]]
y_reg = train_df["usd_pledged"]


#create dummy variables of categorical variables
X_reg = pd.get_dummies(X_reg, columns = ["country","category", "deadline_weekday",	"created_at_weekday","launched_at_weekday",	"deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day"])


#check correlation 
corr_matrix = X_reg.corr(method= 'pearson')
#shows launched year, deadline year and created year are highly correlated to each other hence two of them can be removed from train_df (from line 45) 
#launched_at_month, and launched_at_hr and deadline_month and deadline_hr are correlated so dropped off launched_at_month, and launched_at_hr 
#remove it from train_df to avoid dropping all dummy categories



###--------------------------------------------Feature Analysis-------------------------------------

#split the data into test and training 
from sklearn.model_selection import train_test_split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size = 0.33, random_state = 0)

X_reg_train.reset_index(inplace = True, drop = True)
X_reg_test.reset_index(inplace = True,  drop = True)

y_reg_train = y_reg_train.to_frame()
y_reg_train.reset_index(inplace = True, drop = True)

y_reg_test = y_reg_test.to_frame()
y_reg_test.reset_index(inplace = True,  drop = True)

#getting rid of outliers 
# Create isolation forest model


from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators = 100,  contamination = 0.02, random_state =0) 
pred = iforest.fit_predict(X_reg_train) 
score = iforest.decision_function(X_reg_train)


# Extracting anomalies
from numpy import where
anom_index = where(pred== -1)
values = X_reg_train.iloc[anom_index]

#for i in values_index:
X_reg_train= X_reg_train.drop(anom_index[0], axis = 0)
y_reg_train = y_reg_train.drop(anom_index[0], axis = 0)
#Random Forest 
#from sklearn import preprocessing
#lab_enc = preprocessing.LabelEncoder()
#y_reg = lab_enc.fit_transform(y_reg)

#find feature importance 
from sklearn.ensemble import RandomForestRegressor 
#chose n_estimators = 1000 for higher accuracy 
randomforest = RandomForestRegressor(random_state=0,n_estimators = 300)
#train the model 
model = randomforest.fit(X_reg_train, y_reg_train)
model.feature_importances_
#making dataframe of features along with the values of their importance 
feature_random_forest = pd.DataFrame(list(zip(X_reg.columns,model.feature_importances_)), columns = ['predictor','feature importance'])
#Less important features with importance less than 0.03 are 'created_at_month_12', 'launched_at_weekday_Saturday', 'country_CA', 'category_Robots', 'category_Immersive', 'country_AU', 'category_Flight','category_Spaces', 'deadline_yr_2017', 'deadline_yr_2011', 'country_DE', 'category_Places', 'country_FR', 'category_Makerspaces', 'country_IT', 'category_Others', 'deadline_yr_2010', 'country_Others', 'country_NL', 'country_ES', 'country_IE' ,'country_CH', 'country_NZ', 'country_DK', 'country_SE', 'deadline_yr_2009'

    
### LASSO Feature selection 

#standradize the data first 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

scaler = StandardScaler()
X_lasso_train = scaler.fit_transform(X_reg_train)
y_lasso_train = y_reg_train.iloc[:,0]
X_lasso_test = scaler.fit_transform(X_reg_test)
def lasso(i):
    model = Lasso(alpha=i, max_iter = 10000, random_state = 0)
    model.fit(X_lasso_train, y_lasso_train)
    return model.coef_

lasso_001 = lasso(0.001)
feature_lasso_001= pd.DataFrame(list(zip(X_reg.columns,lasso_001)), columns = ['predictor','coefficient'])
#no predictors with zero coefficients for alpha = 0.001

lasso_005 = lasso(0.005)
feature_lasso_005= pd.DataFrame(list(zip(X_reg.columns,lasso_005)), columns = ['predictor','coefficient'])    
#no predictors with zero coefficients for alpha = 0.005

lasso_01 = lasso(0.01)
feature_lasso_01= pd.DataFrame(list(zip(X_reg.columns,lasso_01)), columns = ['predictor','coefficient'])
#no predictors with zero coefficients for alpha = 0.01

lasso_05 = lasso(0.05)
feature_lasso_05= pd.DataFrame(list(zip(X_reg.columns,lasso_05)), columns = ['predictor','coefficient'])
#predictors with zero coefficients for alpha = 0.05 are deadline_weekday_Friday and created_at_day_(21, 32]
                
lasso_1 = lasso(1.0)
feature_lasso_1= pd.DataFrame(list(zip(X_reg.columns,lasso_1)), columns = ['predictor','coefficient'])
#predictors with zero coefficients for alpha = 1.0 are 'country_AU', 'category_Others','category_Web', 'deadline_weekday_Friday', 'created_at_weekday_Friday', 'deadline_day_(14, 21]', 'deadline_yr_2014','created_at_month_2', 'created_at_day_(21, 32]', 'launched_at_day_(7, 14]'

lasso_5 = lasso(5.0)
feature_lasso_5= pd.DataFrame(list(zip(X_reg.columns,lasso_5)), columns = ['predictor','coefficient'])
#predictors with zero coefficients for alpha = 5.0 are 'country_AU', 'category_Others','category_Web', 'deadline_weekday_Friday', 'created_at_weekday_Friday', 'deadline_day_(14, 21]', 'deadline_yr_2014', 'deadline_hr_(12, 16]', 'created_at_day_(21, 32]', 'launched_at_day_(7, 14]'


lasso_10 = lasso(10.0)
feature_lasso_10= pd.DataFrame(list(zip(X_reg.columns,lasso_10)), columns = ['predictor','coefficient'])
#predictors with zero coefficients for alpha = 10.0 are country_AU, category_Others, category_Web, deadline_weekday_Friday, created_at_weekday_Friday,  deadline_day_(14, 21], deadline_yr_2014, deadline_hr_(16, 20], created_at_month_9, created_at_day_(21, 32], launched_at_day_(7, 14]


##The results vary mostly for alpha = 0.05, 1.0 and 5.0
#There will be three eature lists that will be genrated for regression model

#alpha = 0.05
X_train_reg1 = X_reg_train.drop(columns = ['deadline_weekday_Friday', 'created_at_day_(21, 32]'])
X_test_reg1 =  X_reg_test.drop(columns =['deadline_weekday_Friday', 'created_at_day_(21, 32]'])

#alpha = 1.0
X_train_reg2 = X_reg_train.drop(columns = ['country_AU', 'category_Others','category_Web', 'deadline_weekday_Friday', 'created_at_weekday_Friday', 'deadline_day_(14, 21]', 'deadline_yr_2014', 'created_at_month_2', 'deadline_hr_(12, 16]', 'created_at_day_(21, 32]', 'launched_at_day_(7, 14]'])
X_test_reg2 = X_reg_test.drop(columns = ['country_AU', 'category_Others','category_Web', 'deadline_weekday_Friday', 'created_at_weekday_Friday', 'deadline_day_(14, 21]', 'deadline_yr_2014',  'created_at_month_2','deadline_hr_(12, 16]', 'created_at_day_(21, 32]', 'launched_at_day_(7, 14]'])


#alpha = 10.0
X_train_reg3 = X_reg_train.drop(columns = ['country_AU', 'category_Others', 'category_Web', 'deadline_weekday_Friday', 'created_at_weekday_Friday',  'deadline_day_(14, 21]', 'deadline_yr_2014', 'deadline_hr_(16, 20]', 'created_at_month_9', 'created_at_day_(21, 32]', 'launched_at_day_(7, 14]'])
X_test_reg3 = X_reg_test.drop(columns = ['country_AU', 'category_Others', 'category_Web', 'deadline_weekday_Friday', 'created_at_weekday_Friday',  'deadline_day_(14, 21]', 'deadline_yr_2014', 'deadline_hr_(16, 20]', 'created_at_month_9', 'created_at_day_(21, 32]', 'launched_at_day_(7, 14]'])


#results from random forest feature analysis 

X_train_reg4 = X_reg_train.drop(columns=['country_ES', 'country_CA', 'launched_at_weekday_Saturday','country_IT','category_Musical', 'country_IE', 'country_FR', 'category_Apps','category_Spaces', 'deadline_yr_2010', 'country_CH', 'category_Immersive', 'category_Others', 'country_DK', 'category_Plays', 'category_Experimental', 'category_Makerspaces', 'category_Festivals', 'country_NZ', 'category_Places', 'deadline_yr_2009'])
X_test_reg4 = X_reg_test.drop(columns=['country_ES', 'country_CA', 'launched_at_weekday_Saturday','country_IT','category_Musical', 'country_IE', 'country_FR', 'category_Apps','category_Spaces', 'deadline_yr_2010', 'country_CH', 'category_Immersive', 'category_Others', 'country_DK', 'category_Plays', 'category_Experimental', 'category_Makerspaces', 'category_Festivals', 'country_NZ', 'category_Places', 'deadline_yr_2009'])


y_train_reg1 = y_reg_train.iloc[:,0]
y_test_reg1 =  y_reg_test.iloc[:,0]



###------------------------------------------Regression Model-------------------------------------------------------------


##Simple linear regression 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Run linear regression
def lm(X_train,y_train, X_test, y_test):
    lm1 = LinearRegression()
    model1 = lm1.fit(X_train,y_train)
    y_test_pred_lm1 = model1.predict(X_test)
    mse_lm1 = mean_squared_error(y_test, y_test_pred_lm1)
    return mse_lm1

lm1_mse = lm(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
lm2_mse = lm(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
lm3_mse = lm(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
lm4_mse = lm(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)


print("Linear Model Regression MSE when alpha = 0.05 was used in feature analysis using LASSO = "+str(lm1_mse))
#gives mse of 19865555917.813114
print("Linear Model Regression MSE when alpha = 1 was used in feature analysis using LASSO = "+str(lm2_mse))
print("Linear Model Regression MSE when alpha = 10 was used in feature analysis using LASSO = "+str(lm3_mse))
print("Linear Model Regression MSE with features selected from Random Forest = "+str(lm4_mse))

###Results 
#Linear Model Regression MSE when alpha = 0.05 was used in feature analysis using LASSO = 18798287875.504295
#Linear Model Regression MSE when alpha = 1 was used in feature analysis using LASSO = 18798179325.258255
#Linear Model Regression MSE when alpha = 10 was used in feature analysis using LASSO = 18798079408.60514
#Linear Model Regression MSE with features selected from Random Forest = 18804911159.15919, 18798930802.55926

##Ridge regression 
#trying to find the lowest MSE using different alpha values

#standardize the variables first for Lasso, Ridge and KNN

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std_train1 = scaler.fit_transform(X_train_reg1)
X_std_test1 = scaler.fit_transform(X_test_reg1)
X_std_train2 = scaler.fit_transform(X_train_reg2)
X_std_test2 = scaler.fit_transform(X_test_reg2)
X_std_train3 = scaler.fit_transform(X_train_reg3)
X_std_test3= scaler.fit_transform(X_test_reg3)
X_std_train4 = scaler.fit_transform(X_train_reg4)
X_std_test4 = scaler.fit_transform(X_test_reg4)



from sklearn.linear_model import Ridge

def ridge(X_train,y_train, X_test, y_test):
    ridge_alpha = {}
    for i in range (2,10):
        ridge1 = Ridge(alpha=i, random_state = 0)
        model2 = ridge1.fit(X_train,y_train)
        y_test_pred_ridge1 = model2.predict(X_test)
        ridge_alpha[i] = mean_squared_error(y_test, y_test_pred_ridge1)
    return min(ridge_alpha.values())

ridge1_mse = ridge(X_std_train1, y_train_reg1, X_std_test1,y_test_reg1)
ridge2_mse = ridge(X_std_train2, y_train_reg1, X_std_test2,y_test_reg1)
ridge3_mse = ridge(X_std_train3, y_train_reg1, X_std_test3,y_test_reg1)
ridge4_mse = ridge(X_std_train4, y_train_reg1, X_std_test4,y_test_reg1)



print("Ridge Model Regression MSE when alpha = 0.05 was used in feature analysis using LASSO = "+str(ridge1_mse))
print("Ridge Model Regression MSE when alpha = 1 was used in feature analysis using LASSO = "+str(ridge2_mse))
print("Ridge Model Regression MSE when alpha = 10 was used in feature analysis using LASSO = "+str(ridge3_mse))
print("Ridge Model Regression MSE with features selected from Random Forest = "+str(ridge4_mse))

#results
#Ridge Model Regression MSE when alpha = 0.05 was used in feature analysis using LASSO = 18794550698.98613
#Ridge Model Regression MSE when alpha = 1 was used in feature analysis using LASSO = 18794846876.37129
#Ridge Model Regression MSE when alpha = 10 was used in feature analysis using LASSO = 18794779459.261013
#Ridge Model Regression MSE with features selected from Random Forest = 18795255149.1137  


##Lasso
def lasso(X_train,y_train, X_test, y_test):
    lasso_alpha = {}
    for i in range (1,10):
        lasso1 = Lasso(alpha=i, random_state = 0)
        model3 = lasso1.fit(X_train,y_train)
        y_test_pred_lasso1 = model3.predict(X_test)
        lasso_alpha[i] = mean_squared_error(y_test, y_test_pred_lasso1)
    return min(lasso_alpha.values())

lasso1_mse = lasso(X_std_train1, y_train_reg1, X_std_test1,y_test_reg1)
lasso2_mse = lasso(X_std_train2, y_train_reg1, X_std_test2,y_test_reg1)
lasso3_mse = lasso(X_std_train3, y_train_reg1, X_std_test3,y_test_reg1)
lasso4_mse = lasso(X_std_train4, y_train_reg1, X_std_test4,y_test_reg1)

print("Lasso Model Regression MSE when alpha = 0.05 was used in feature analysis using LASSO = "+str(lasso1_mse))
print("Lasso Model Regression MSE when alpha = 1 was used in feature analysis using LASSO = "+str(lasso2_mse))
print("Lasso Model Regression MSE when alpha = 5 was used in feature analysis using LASSO = "+str(lasso3_mse))
print("Lasso Model Regression MSE with features selected from Random Forest = "+str(lasso4_mse))

#results
#Lasso Model Regression MSE when alpha = 0.05 was used in feature analysis using LASSO = 18793530055.710464
#Lasso Model Regression MSE when alpha = 1 was used in feature analysis using LASSO = 18793544617.89055
#Lasso Model Regression MSE when alpha = 5 was used in feature analysis using LASSO = 18793537259.525406
#Lasso Model Regression MSE with features selected from Random Forest = 18794210916.776928

##KNN
from sklearn.neighbors import KNeighborsRegressor

# Run K-NN
def knn(X_train,y_train, X_test, y_test): 
    knn_neighbor = {}
    n_neighbor = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
    for i in n_neighbor:
        knn = KNeighborsRegressor(n_neighbors=i)
        model5 = knn.fit(X_train,y_train)
        # Using the model to predict the results based on the test dataset
        y_test_pred_knn = model5.predict(X_test)
        knn_neighbor[i] = mean_squared_error(y_test, y_test_pred_knn)
    return min(knn_neighbor.values())

knn1_mse = knn(X_std_train1, y_train_reg1, X_std_test1,y_test_reg1)
knn2_mse = knn(X_std_train2, y_train_reg1, X_std_test2,y_test_reg1)
knn3_mse = knn(X_std_train3, y_train_reg1, X_std_test3,y_test_reg1)
knn4_mse = knn(X_std_train4, y_train_reg1, X_std_test4,y_test_reg1)

print("MSE from KNN when alpha = 0.05 was used in feature analysis using LASSO = "+str(knn1_mse))
print(" MSE from KNN when alpha = 1 was used in feature analysis using LASSO = "+str(knn2_mse))
print("MSE from KNN when alpha = 5 was used in feature analysis using LASSO = "+str(knn3_mse))
print("MSE from KNN when alpha = 5 was used in feature analysis using LASSO = "+str(knn4_mse))


#MSE from KNN when alpha = 0.05 was used in feature analysis using LASSO = 18780434027.540524
#MSE from KNN when alpha = 1 was used in feature analysis using LASSO = 18979190542.5954
#MSE from KNN when alpha = 5 was used in feature analysis using LASSO = 18992185029.228382
#MSE from KNN when alpha = 5 was used in feature analysis using LASSO = 18992185029.228382


##CART wasn't used as Random Forest builds up on CART and gives better results 

##Random Forest 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Run Random Forest
n_estimators = [100,200,300, 500,1000]
max_features = ['auto', 'sqrt', 'log2']
min_samples_leaf = [100, 200, 300, 400]
max_depth = [5,10,20]
min_samples_split = [2,4,6,8]
# Run Random Forest
def rf_mse(X_train,y_train, X_test, y_test):
    rf_mse1 = {}
    for i in n_estimators: 
        random_forest = RandomForestRegressor(n_estimators = i, random_state = 0)
        model6 = random_forest.fit(X_train,y_train)
        y_test_pred_rf1 = model6.predict(X_test)
        rf_mse1[i] = mean_squared_error(y_test, y_test_pred_rf1)
    return rf_mse1
rf1_mse = rf_mse(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
rf3_mse = rf_mse(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
rf2_mse = rf_mse(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
rf4_mse = rf_mse(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)
#Best results if n_estimators is 1000 for featurelist1
#best MSE:   17577245273.968544 
#all results show 1000 is best for estimator

def rf_features(X_train,y_train, X_test, y_test):
    rf_mse2 = {}
    for i in max_features: 
        random_forest1 = RandomForestRegressor(random_state = 0, max_features = i)
        model7 = random_forest1.fit(X_train,y_train)
        y_test_pred_rf2 = model7.predict(X_test)
        rf_mse2[i] = mean_squared_error(y_test, y_test_pred_rf2)
    return rf_mse2
rf5_mse = rf_features(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
rf6_mse = rf_features(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
rf7_mse = rf_features(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
rf8_mse = rf_features(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)

# 'sqrt': 17306373916.425625,
#sqrt best and feature list 1 

def rf_leaf(X_train,y_train, X_test, y_test):
    rf_mse3 = {}
    for i in min_samples_leaf: 
        random_forest2 = RandomForestRegressor(random_state = 0, min_samples_leaf = i)
        model8 = random_forest2.fit(X_train,y_train)
        y_test_pred_rf3 = model8.predict(X_test)
        rf_mse3[i] = mean_squared_error(y_test, y_test_pred_rf3)
    return rf_mse3
rf9_mse = rf_leaf(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
rf10_mse = rf_leaf(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
rf11_mse = rf_leaf(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
rf12_mse = rf_leaf(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)
# 100 is the best for feature list 1 and 3

def rf_depth(X_train,y_train, X_test, y_test):
    rf_mse4 = {}
    for i in max_depth: 
        random_forest3 = RandomForestRegressor(random_state = 0, max_depth = i)
        model9 = random_forest3.fit(X_train,y_train)
        y_test_pred_rf4 = model9.predict(X_test)
        rf_mse4[i] = mean_squared_error(y_test, y_test_pred_rf4)
    return rf_mse4
rf13_mse = rf_depth(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
rf14_mse = rf_depth(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
rf15_mse = rf_depth(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
rf16_mse = rf_depth(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)
#10 for feature list 1 is the best 
#MSE: 17702345482.0385
    
def rf_split(X_train,y_train, X_test, y_test):
    rf_mse5 = {}
    for i in min_samples_split: 
        random_forest4 = RandomForestRegressor(random_state = 0, min_samples_split = i)
        model10 = random_forest4.fit(X_train,y_train)
        y_test_pred_rf5 = model10.predict(X_test)
        rf_mse5[i] = mean_squared_error(y_test, y_test_pred_rf5)
    return rf_mse5
rf17_mse = rf_split(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
rf18_mse = rf_split(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
rf19_mse = rf_split(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
rf20_mse = rf_split(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)
# 2 is the best for feature list 1: 17799925235.76345,
    

##best model 
random_forest5 = RandomForestRegressor(n_estimators = 100, random_state = 0, max_features = "sqrt")
model11 = random_forest5.fit(X_train_reg1,y_train_reg1)
y_test_pred_rf6 = model11.predict(X_test_reg1)
rf_mse6 = mean_squared_error(y_test_reg1, y_test_pred_rf6)
print(rf_mse6)
#17306373916.425625

##Grid Search CV took too long to run, hence, it wasn't used 
'''
grid_param = { 'max_features': max_features,'max_depth': max_depth, 'min_samples_split' : min_samples_split, 'min_samples_leaf': min_samples_leaf}
rf = RandomForestRegressor(random_state=0)
clf = GridSearchCV(estimator = rf, cv=5, param_grid= grid_param, n_jobs = -1)
model6 = clf.fit(X_train_reg1,y_train_reg1)
y_test_pred_rf = model6.predict(X_test_reg1)
best_params = model6.best_params_
'''

###Gradient Boosting Regressor 
from sklearn.ensemble import GradientBoostingRegressor


def gbt_est(X_train, y_train, X_test, y_test):
    gbt_mse1= {}
    n_estimators1 = [10,20,30]
    for i in n_estimators1:
        gbt1 = GradientBoostingRegressor(n_estimators= i,random_state=0)
        model12 = gbt1.fit(X_train,y_train)
        y_test_pred_gbt1 = model12.predict(X_test)
        gbt_mse1[i] = mean_squared_error(y_test, y_test_pred_gbt1)
    return gbt_mse1

gbt1_mse = gbt_est(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
gbt2_mse  = gbt_est(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
gbt3_mse  = gbt_est(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
gbt4_mse =  gbt_est(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)
#20 is better for feature list 2 and 3: 19555787647.599613

def gbt_rate(X_train, y_train, X_test, y_test):
    gbt_mse2= {}
    learning_rates = [0.001, 0.01,0.1]
    for i in learning_rates:
        gbt2 = GradientBoostingRegressor(learning_rate = i,random_state=0)
        model13 = gbt2.fit(X_train,y_train)
        y_test_pred_gbt2 = model13.predict(X_test)
        gbt_mse2[i] = mean_squared_error(y_test, y_test_pred_gbt2)
    return gbt_mse2
gbt5_mse = gbt_rate(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
gbt6_mse  = gbt_rate(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
gbt7_mse  = gbt_rate(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
gbt8_mse =  gbt_rate(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)
#best is 0.01: 18608180026.552933

def gbt_features(X_train, y_train, X_test, y_test):
    gbt_mse4 = {}
    max_features1 = ['auto', 'sqrt', 'log2']
    for i in max_features1:
        gbt4 = GradientBoostingRegressor(max_features = i,random_state=0)
        model14 = gbt4.fit(X_train,y_train)
        y_test_pred_gbt4 = model14.predict(X_test)
        gbt_mse4[i] = mean_squared_error(y_test, y_test_pred_gbt4)
    return gbt_mse4
gbt9_mse = gbt_features(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
gbt10_mse  = gbt_features(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
gbt11_mse  = gbt_features(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
gbt12_mse =  gbt_features(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)
# 'sqrt': 17927130721.394497 for feature list 3

def gbt_depth(X_train, y_train, X_test, y_test):
    gbt_mse5 = {}
    max_depth1 = [5,10,20]
    for i in max_depth1:
        gbt5 = GradientBoostingRegressor(max_depth = i,random_state=0)
        model16 = gbt5.fit(X_train,y_train)
        y_test_pred_gbt5 = model16.predict(X_test)
        gbt_mse5[i] = mean_squared_error(y_test, y_test_pred_gbt5)
    return gbt_mse5

gbt13_mse = gbt_depth(X_train_reg1, y_train_reg1, X_test_reg1,y_test_reg1)
gbt14_mse  =gbt_depth(X_train_reg2, y_train_reg1, X_test_reg2,y_test_reg1)
gbt15_mse  = gbt_depth(X_train_reg3, y_train_reg1, X_test_reg3,y_test_reg1)
gbt16_mse = gbt_depth(X_train_reg4, y_train_reg1, X_test_reg4,y_test_reg1)
# '5 for feature list 4 17468622067.575798

##Best model for GBT 
gbt6 = GradientBoostingRegressor(random_state=0, max_depth = 5)
model17 = gbt6.fit(X_train_reg4,y_train_reg1)
y_test_pred_gbt6 = model17.predict(X_test_reg4)
gbt_mse_test = mean_squared_error(y_test_reg1, y_test_pred_gbt6)
gbt_mse_test
#result = 17468622067.575798
'''
grid_param1 = {'larning rates': learning_rates1, 'n_estimators': n_estimators1, 'max_features': max_features1,'max_depth': max_depth1, 'min_samples_split' : min_samples_split1, 'min_samples_leaf': min_samples_leaf1, "subsample": subsample}

gbt = GradientBoostingRegressor(random_state=0)
gbr = GridSearchCV(estimator = gbt, cv=5, param_grid= grid_param1)
model10 = gbr.fit(X_train_reg1,y_train_reg1)
y_test_pred_gbt = model10.predict(X_test_reg1)
best_params_gbt = model10.best_params_
print(model10.best_params_)
'''
###ANN
from sklearn.neural_network import MLPRegressor 

standardizer = StandardScaler()
X_train_ann1 = standardizer.fit_transform(X_train_reg1)
X_test_ann1 = standardizer.fit_transform(X_test_reg1)
X_train_ann2 = standardizer.fit_transform(X_train_reg2)
X_test_ann2 = standardizer.fit_transform(X_test_reg2)
X_train_ann3  = standardizer.fit_transform(X_train_reg3)
X_test_ann3 = standardizer.fit_transform(X_test_reg3)
X_train_ann4  = standardizer.fit_transform(X_train_reg4)
X_test_ann4= standardizer.fit_transform(X_test_reg4)
# Run ANN

def mlp(X_train, y_train, X_test, y_test):
    ann_mse = {}
    for i in range(1,21):
        ann1 = MLPRegressor(hidden_layer_sizes= i,max_iter=1000, random_state = 0)
        model21 = ann1.fit(X_train,y_train)
        y_test_pred_ann1 = model21.predict(X_test)
        ann_mse[i] = mean_squared_error(y_test, y_test_pred_ann1)
    return ann_mse

mlp_mse1 = mlp(X_train_ann1, y_train_reg1, X_test_ann1, y_test_reg1)
mlp_mse2 = mlp(X_train_ann2, y_train_reg1, X_test_ann2, y_test_reg1)
mlp_mse3 = mlp(X_train_ann3, y_train_reg1, X_test_ann3, y_test_reg1)
mlp_mse4 = mlp(X_train_ann4, y_train_reg1, X_test_ann4, y_test_reg1)
#did not give good results as compared to other methods 




###------------------------------------CLASSIFICATION MODEL-----------------------------------------------

###Feature Analysis 

X_class = train_df[["goal","country","category", "name_len_clean", "blurb_len_clean","deadline_weekday", "created_at_weekday","launched_at_weekday","deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day","create_to_launch_days", "launch_to_deadline_days"]]
y_class = train_df["state"]


#create dummy variables of categorical variables
X_class = pd.get_dummies(X_class, columns = ["country","category", "deadline_weekday",	"created_at_weekday","launched_at_weekday",	"deadline_month","deadline_day","deadline_yr","deadline_hr", "created_at_month","created_at_day","created_at_hr","launched_at_day"])


#check correlation 
corr_matrix = X_class.corr(method= 'pearson')
#shows launched year, deadline year and created year are highly correlated to each other hence two of them can be removed from train_df (from line 45) 
#launched_at_month, and launched_at_hr and deadline_month and deadline_hr are correlated so dropped off launched_at_month, and launched_at_hr 
#remove it from train_df to avoid dropping all dummy categories



#split the data into test and training 
from sklearn.model_selection import train_test_split
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size = 0.33, random_state = 0)

X_class_train.reset_index(inplace = True, drop = True)
X_class_test.reset_index(inplace = True,  drop = True)

y_class_train = y_class_train.to_frame()
y_class_train.reset_index(inplace = True, drop = True)

y_class_test = y_class_test.to_frame()
y_class_test.reset_index(inplace = True,  drop = True)

#getting rid of outliers 
# Create isolation forest model

from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators = 100,  contamination = 0.02, random_state =0) 
pred = iforest.fit_predict(X_class_train) 
score = iforest.decision_function(X_class_train)

# Extracting anomalies
from numpy import where
anom_index = where(pred== -1)
values = X_class_train.iloc[anom_index]

#for i in values_index:
X_class_train= X_class_train.drop(anom_index[0], axis = 0)
y_class_train = y_class_train.drop(anom_index[0], axis = 0)


from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
y_class_train = lab_enc.fit_transform(y_class_train)
y_class_test = lab_enc.fit_transform(y_class_test)

#find feature importance 
from sklearn.ensemble import RandomForestClassifier
randomforest_c1 = RandomForestClassifier(random_state=0)
#train the model 
model_c1 = randomforest_c1.fit(X_class_train, y_class_train)
features_rf_c1 = model_c1.feature_importances_
#making dataframe of features along with the values of their importance 
feature_random_forest_c1 = pd.DataFrame(list(zip(X_class.columns,model_c1.feature_importances_)), columns = ['predictor','feature importance'])


X_train_rf = X_class_train.drop(columns = ['deadline_yr_2011','country_DE', 'category_Places', 'country_FR', 'category_Makerspaces', 'country_IT', 'category_Others', 'deadline_yr_2010','country_Others', 'country_NL', 'country_ES', 'country_IE', 'country_CH','country_NZ','country_DK' ,'country_SE', 'deadline_yr_2009'])
X_test_rf = X_class_test.drop(columns = ['deadline_yr_2011','country_DE', 'category_Places', 'country_FR', 'category_Makerspaces', 'country_IT', 'category_Others', 'deadline_yr_2010','country_Others', 'country_NL', 'country_ES', 'country_IE', 'country_CH','country_NZ','country_DK' ,'country_SE', 'deadline_yr_2009'])

X_train_rf2 = X_class_train.drop(columns = ['deadline_yr_2011','country_DE', 'category_Places', 'country_FR', 'category_Makerspaces', 'country_IT', 'category_Others', 'deadline_yr_2010','country_Others', 'country_NL', 'country_ES', 'country_IE', 'country_CH','country_NZ','country_DK' ,'country_SE', 'deadline_yr_2009', 'category_Robots', 'category_Immersive', 'country_AU', 'category_Flight', 'category_Spaces', 'deadline_yr_2017'])
X_test_rf2 = X_class_test.drop(columns = ['deadline_yr_2011','country_DE', 'category_Places', 'country_FR', 'category_Makerspaces', 'country_IT', 'category_Others', 'deadline_yr_2010','country_Others', 'country_NL', 'country_ES', 'country_IE', 'country_CH','country_NZ','country_DK' ,'country_SE', 'deadline_yr_2009', 'category_Robots', 'category_Immersive', 'country_AU', 'category_Flight', 'category_Spaces', 'deadline_yr_2017'])
# "launched_at_weekday_Saturday","country_CA", "created_at_month_12", created_at_month_2 category_Sound deadline_month_1 deadline_month_2 category_Wearables launched_at_weekday_Sunday category_Apps, deadline_yr_2012


X_train_rf2.columns
from sklearn.metrics import accuracy_score

#Running different random forest models  with eliminating fetaures 
randomforest_c2 = RandomForestClassifier()
model_c2 = randomforest_c2.fit(X_train_rf, y_class_train)
y_test_pred_c2 = model_c2.predict(X_test_rf)
acc_rf_c1 = accuracy_score(y_class_test,y_test_pred_c2)
#accuracy score of 0.732679599232573

randomforest_c3 = RandomForestClassifier()
model_c3 = randomforest_c3.fit(X_train_rf2, y_class_train)
y_test_pred_c3 = model_c3.predict(X_test_rf2)
acc_rf_c2 = accuracy_score(y_class_test,y_test_pred_c3)
# accuracy score of  0.7365167341718184

n_estimators_class = [100, 300, 500, 800, 1200]
max_depth_class = [5, 8, 15, 25, 30]
min_samples_split_class = [2, 5, 10, 15, 100]
min_samples_leaf_class = [1, 2, 5, 10] 


grid_parameters = dict(n_estimators = n_estimators_class, max_depth = max_depth_class, min_samples_split = min_samples_split_class, min_samples_leaf = min_samples_leaf_class)
forest = RandomForestClassifier(random_state=0)
gridF = GridSearchCV(forest,grid_parameters, cv = 3, verbose = 1, n_jobs = -1)
best_rf = gridF.fit(X_train_rf2, y_class_train)
best_params_rf = best_rf.best_params_
best_score_rf = best_rf.best_score_
#gives 0.7429281293808265
#gives params as  n_estimators =300 , max_depth = 40 , min_samples_leaf = 1, min_samples_split = 5


#using grid search Cv results
#final model for Random Forest  
randomforest_class = RandomForestClassifier(random_state = 0, n_estimators =300 , max_depth = 40 , min_samples_leaf = 1, min_samples_split = 5)
model_rf = randomforest_class.fit(X_train_rf2, y_class_train)
y_test_pred_c4 = model_rf.predict(X_test_rf2)
acc_rf_class= accuracy_score(y_class_test,y_test_pred_c4) 
print("Accuracy from Random Forest Classifier:", acc_rf_class)
# gives 0.7414197399275207

'''
### LASSO Feature selection (results not used since Random Forest feature selection technique worked better)

#standradize the data first 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_lasso_train_clas = scaler.fit_transform(X_class_train)
y_lasso_train_clas = y_class_train
X_lasso_test_clas = scaler.fit(X_class_test)
from sklearn.linear_model import Lasso
def lasso_class(i):
    model_c2 = Lasso(alpha=i, max_iter = 10000)
    model_c2.fit(X_lasso_train_clas, y_lasso_train_clas)
    return model_c2.coef_

lasso_001_clas = lasso_class(0.001)
feature_lasso_001_clas = pd.DataFrame(list(zip(X_class.columns,lasso_001_clas)), columns = ['predictor','coefficient'])
#no predictors with zero coefficients for alpha = 0.001


lasso_005_clas = lasso_class(0.005)
feature_lasso_005_clas = pd.DataFrame(list(zip(X_class.columns,lasso_005_clas)), columns = ['predictor','coefficient'])  
#no predictors with zero coefficients for alpha = 0.005

lasso_01_clas = lasso(0.01)
feature_lasso_0_clas1= pd.DataFrame(list(zip(X_class_train.columns,lasso_01_clas)), columns = ['predictor','coefficient'])
#predictors with zero coefficients for alpha = 0.01 are deadline_weekday_Wednesday 

lasso_05_clas = lasso(0.05)
feature_lasso_05_clas= pd.DataFrame(list(zip(X_class_train.columns,lasso_05_clas)), columns = ['predictor','coefficient'])
#predictors with zero coefficients for alpha = 0.05 are deadline_weekday_Wednesday and created_at_day_(21, 32]

lasso_1_clas = lasso(1.0)
feature_lasso_1_clas = pd.DataFrame(list(zip(X_class_train.columns,lasso_1_clas)), columns = ['predictor','coefficient'])
##predictors with zero coefficients for alpha = 1.0 are category_Web, deadline_weekday_Wednesday,created_at_weekday_Friday, deadline_month_6, deadline_day_(14, 21], deadline_yr_2014, created_at_month_9, created_at_day_(21, 32], launched_at_day_(0, 7]

lasso_5_clas = lasso(5.0)
feature_lasso_5_clas = pd.DataFrame(list(zip(X_class_train.columns,lasso_5_clas)), columns = ['predictor','coefficient'])
##predictors with zero coefficients for alpha = 5.0 are country_CH,category_Web, deadline_weekday_Wednesday, created_at_weekday_Friday, deadline_month_6, deadline_day_(14, 21], deadline_yr_2014, deadline_hr_(16, 20], created_at_month_2, created_at_month_9, created_at_day_(21, 32], and launched_at_day_(0, 7]

lasso_10_clas = lasso(10.0)
feature_lasso_10_clas = pd.DataFrame(list(zip(X_class_train.columns,lasso_10_clas)), columns = ['predictor','coefficient'])
#predictors with zero coefficients for alpha = 10.0 are country_CH, category_Web, deadline_weekday_Wednesday, created_at_weekday_Friday, deadline_month_6, deadline_day_(14, 21], deadline_yr_2014, created_at_month_2, created_at_month_9, created_at_day_(21, 32], and launched_at_day_(0, 7]
'''
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

learning_rates_gbt = [0.001,0.005,0.01,0.05,0.1]
n_estimators_gbt = [100,150,200]
max_depth_gbt = [5, 8, 15, 25, 30]
min_samples_split_gbt = [2, 5, 10, 15, 100]

grid_parameters_gbt = dict(n_estimators = n_estimators_gbt, max_depth = max_depth_gbt, min_samples_split = min_samples_split_gbt, learning_rate =learning_rates_gbt)
gbt_class = GradientBoostingClassifier(random_state=0)
model_gbt_class = GridSearchCV(gbt_class,grid_parameters_gbt, cv = 3, verbose = 1, n_jobs = -1)
best_gbt = model_gbt_class.fit(X_train_rf2, y_class_train)
best_params_gbt = best_gbt.best_params_
best_score_gbt = best_gbt.best_score_

acc_gbt_c1 = {}
for i in n_estimators_gbt:
    gbt_class = GradientBoostingClassifier(random_state=0, n_estimators = i)
    model_gbt = gbt_class.fit(X_train_rf2, y_class_train)
    y_test_pred_c1 = model_gbt.predict(X_test_rf2)
    acc_gbt_c1[i]= accuracy_score(y_class_test,y_test_pred_c1) 
print(acc_gbt_c1)
#best accuracy when n_estimators = 200: 0.7529311447452569)


acc_gbt_c2 = {}
for i in learning_rates_gbt:
    gbt_class1 = GradientBoostingClassifier(random_state=0, learning_rate = i)
    model_gbt1 = gbt_class1.fit(X_train_rf2, y_class_train)
    y_test_pred_c2 = model_gbt1.predict(X_test_rf2)
    acc_gbt_c2[i]= accuracy_score(y_class_test,y_test_pred_c2) 
print(acc_gbt_c2)
# Best ar 0.1: 0.75037305478576
#leave at default 

acc_gbt_c3 = {}
for i in max_depth_gbt:
    gbt_class2 = GradientBoostingClassifier(random_state=0,max_depth = i)
    model_gbt2 = gbt_class2.fit(X_train_rf2, y_class_train)
    y_test_pred_c3 = model_gbt2.predict(X_test_rf2)
    acc_gbt_c3[i]= accuracy_score(y_class_test,y_test_pred_c3)     
print(acc_gbt_c3)
#max_depth= 5: 0.75037305478576


acc_gbt_c4 = {}
for i in min_samples_split_gbt:
    gbt_class3 = GradientBoostingClassifier(random_state=0,min_samples_split = i)
    model_gbt3 = gbt_class3.fit(X_train_rf2, y_class_train)
    y_test_pred_c4 = model_gbt3.predict(X_test_rf2)
    acc_gbt_c4[i]= accuracy_score(y_class_test,y_test_pred_c4)   
print(acc_gbt_c4)
#min_sample_split = 2: 0.75037305478576
    
#final model     
gbt_class = GradientBoostingClassifier(random_state=0, n_estimators = 200)
model_gbt = gbt_class.fit(X_train_rf2, y_class_train)
y_test_pred_c1 = model_gbt.predict(X_test_rf2)
acc_gbt_class= accuracy_score(y_class_test,y_test_pred_c1) 
print("Accuracy from GBT Classier:", acc_gbt_class)
#answer of 0.7529311447452569)
#Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_class_test,y_test_pred_c1) )

#Calculate precision
from sklearn.metrics import precision_score
print("Precision Score", precision_score(y_class_test,y_test_pred_c1) )

#Calculate recall
from sklearn.metrics import recall_score
print("Recall Score", recall_score(y_class_test,y_test_pred_c1))

#Calculate f1 score
from sklearn.metrics import f1_score
print("F1 Score", f1_score(y_class_test,y_test_pred_c1))


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

standardizer = StandardScaler()
X_train_knn_class = standardizer.fit_transform(X_train_rf2)
X_test_knn_class = standardizer.fit_transform(X_test_rf2)
acc_knn_class = {}
for i in range (1,21):
    knn_class = KNeighborsClassifier(n_neighbors=i)
    model_knn = knn_class.fit(X_train_knn_class,y_class_train)
    y_test_pred_knn = model_knn.predict(X_test_knn_class)
    acc_knn_class[i] = accuracy_score(y_class_test, y_test_pred_knn)
print(max(acc_knn_class.values()))
# 13: 0.6881261991046685,

# Run the model
from sklearn.linear_model import LogisticRegression
logreg_class = LogisticRegression(random_state = 0, max_iter = 10000)
model_logreg = logreg_class.fit(X_train_rf2,y_class_train)
y_test_pred_logreg = model_logreg.predict(X_test_rf2)
acc_logreg_class = accuracy_score(y_class_test, y_test_pred_logreg)
print(acc_logreg_class)
#0.7203154977616713


acc_mlp_class = {}
from sklearn.neural_network import MLPClassifier
for i in range(2,21):
    mlp_class= MLPClassifier(hidden_layer_sizes=i,max_iter=1000,random_state=0)
    model_mlp = mlp_class.fit(X_train_rf2,y_class_train)
    y_test_pred_mlp = model_mlp.predict(X_test_rf2)
    acc_mlp_class[i] = accuracy_score(y_class_test, y_test_pred_mlp)
print(max(acc_mlp_class.values()))
# 16: 0.7141334470262204 


###------------------------------------Clustering------------------------------------------------------------------------------------------------------------------------------------------


X_clust = kickstarter_train_df[["goal","name_len_clean","blurb_len_clean","static_usd_rate", "launch_to_deadline_days", "state", "backers_count", "staff_pick"]]
X_clust = X_clust.dropna()
X_clust.reset_index(inplace = True, drop = True)

X_clust['goal'] = X_clust['goal']*X_clust['static_usd_rate']
X_clust = X_clust.drop(columns = ["static_usd_rate"])

from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
X_clust["state"] = lab_enc.fit_transform(X_clust["state"])
X_clust["staff_pick"] = lab_enc.fit_transform(X_clust["staff_pick"])

#reset the index
X_clust.reset_index(inplace = True, drop = True)


#getting rid of outliers 
# Create isolation forest model

from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators = 100,  contamination = 0.05, random_state =0) 
pred = iforest.fit_predict(X_clust) 
score = iforest.decision_function(X_clust)

# Extracting anomalies
from numpy import where
anom_index = where(pred== -1)
values = X_clust.iloc[anom_index]

#for i in values_index:
X_clust = X_clust.drop(anom_index[0], axis = 0)

#standardizing the data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std_clust = scaler.fit_transform(X_clust)
X_std_clust = pd.DataFrame(X_std_clust,  columns=['goal', 'name_len_clean', 'blurb_len_clean', 'launch_to_deadline_days','state', 'backers_count', 'staff_pick'])

#K Means 
from sklearn.cluster import KMeans
withinss = []
for i in range (2,10):
    kmeans = KMeans(n_clusters=i)
    model_clust = kmeans.fit(X_std_clust)
    withinss.append(model_clust.inertia_)
from matplotlib import pyplot
pyplot.plot([2,3,4,5,6,7,8,9],withinss)

#n_clusters = 7

kmeans1 = KMeans(n_clusters=7)
model_clust1 = kmeans1.fit(X_std_clust)
labels = model_clust1.predict(X_std_clust)
X_clust['Cluster'] = labels
X_std_clust['Cluster'] = labels

from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_std_clust,labels)
df = pd.DataFrame({'label':labels,'silhouette':silhouette})
print('Average Silhouette Score for Cluster 0: ',numpy.average(df[df['label'] == 0].silhouette))
print('Average Silhouette Score for Cluster 1: ',numpy.average(df[df['label'] == 1].silhouette))
print('Average Silhouette Score for Cluster 2: ',numpy.average(df[df['label'] == 2].silhouette))
print('Average Silhouette Score for Cluster 3: ',numpy.average(df[df['label'] == 3].silhouette))
print('Average Silhouette Score for Cluster 4 ',numpy.average(df[df['label'] == 4].silhouette))
print('Average Silhouette Score for Cluster 5: ',numpy.average(df[df['label'] == 5].silhouette))
print('Average Silhouette Score for Cluster 6: ',numpy.average(df[df['label'] == 6].silhouette))


df_nor_melt = pd.melt(X_std_clust,
                      id_vars=['Cluster'],
                      value_vars=['goal', 'name_len_clean', 'blurb_len_clean', 'launch_to_deadline_days','state', 'backers_count', 'staff_pick'],
                      var_name='Attribute',
                      value_name='Value')

fig = plt.figure(figsize=(20,10))
plot = sns.lineplot('Attribute', 'Value', hue='Cluster', color = "viridis", data=df_nor_melt)
sns.color_palette()


#centers and inverse standardized centers of the clusters
center = model_clust1.cluster_centers_
print(center)
center_df = pd.DataFrame(center, columns = X_clust.columns)
centroid = pd.DataFrame(scaler.inverse_transform(center), columns = X_clust.columns)

