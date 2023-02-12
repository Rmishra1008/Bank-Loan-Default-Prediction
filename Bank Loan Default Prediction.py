#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import dataset

raw = pd.read_csv(r"D:\MSBA Study Materials\Projects\Machine Learning\Loan_Default (1).csv")


# ## Exploratory Data Analysis
# We will first take a look at the dataset then examine if there are duplicants or missing values in the dataset.

# In[2]:


raw.head()


# In[3]:


# ID, year can be dropped as they do not affect the loan status
raw.drop(['ID','year'],axis=1,inplace=True)


# In[4]:


# find if there are duplicants
print("Dataset contains ",raw.duplicated().sum()," lines of dulicated records.")
# print(raw[raw.duplicated()].isna().sum())
# raw[raw.duplicated()]


# In[5]:


# find missing values
def na_pct(df, height=600):
  '''
  pass a dataframe and make a donut plot showing the percentage of unique rows missing data
  and plot missing values as bars and appends text of missing values on the bar
  '''
  pd.Series([len(df)-len(df.dropna()),len(df.dropna())]).plot(kind='pie',
                                                                          title='Proportion of Rows with Missing Values',
                                                                          #  explode=(0,0.1),
                                                                          autopct=lambda p:f'{p:.2f}%, {int(p*len(df)/100)}',
                                                                          labels=['',''],
                                                                          ylabel='',
                                                                          cmap='coolwarm_r'
                                                                          )
  plt.style.use('fast')
  centre_circle = plt.Circle((0,0),0.70,fc='white')
  fig = plt.gcf()
  fig.gca().add_artist(centre_circle)
  plt.legend(['Missing','Filled'])
  plt.show()

  f,ax = plt.subplots(1,1,figsize=(25,10))
  na_df = pd.DataFrame(df.isna().sum()).reset_index()
  na_df.columns=['Columns','Missing Values']
  sns.barplot(data=na_df, x='Columns',y='Missing Values',edgecolor='white', linewidth=3,palette='Pastel1')
  ax.tick_params('x',labelrotation=90)
  for i in na_df.index:
      ax.text(x=i,y=na_df['Missing Values'][i]+height, s=na_df['Missing Values'][i],ha="center", va="center", color="black",size=14)
  f.suptitle('Plot Of Missing Value',y=0.98,ha='center',va='center',size=15, weight=150)
  
  plt.show()
  plt.figure(figsize=(16,9))
  sns.heatmap(df.isnull(), yticklabels = False)

na_pct(raw)


# The missing value comprises over $1/3$ of the total valume, it's a significant portion to discard, therefore, we will impute the missing value with the mode in the corresponding feature.

# In[6]:


# find target classes distribution
def target_pct(df):
  # pass a dataframe and make a donut plot showing the percentage of missing data
  pd.Series(df.Status.value_counts()).plot(kind='pie',
                                                                          title='Proportions of Loan Status',
                                                                          #  explode=(0,0.1),
                                                                          autopct=lambda p:f'{p:.2f}%, {int(p*len(df)/100)}',
                                                                          labels=['',''],
                                                                          ylabel='',
                                                                          cmap='coolwarm'
                                                                          )
  font = {'family' : 'DejaVu Sans',
  'weight' : 'bold',
  'size'   : 15}
  plt.rc('font', **font)  
  plt.style.use('fast')
  centre_circle = plt.Circle((0,0),0.70,fc='white')
  fig = plt.gcf()
  fig.gca().add_artist(centre_circle)
  plt.legend(['Paid-off (0)','Default (1)'])
  plt.show()

target_pct(raw)


# In[7]:


raw[raw.columns[raw.isna().sum()>0]].dtypes


# In[8]:


def plot_categorical(data,height):
    '''
    This function plots categorical variables as bar charts wit
    h the % weight of each value on the bar
    '''
    cat_col = [col for col in data.columns if data[col].dtypes=='object']
    f,ax = plt.subplots(int(len(cat_col)/7),7, figsize=(33,20))
    for i in range(len(cat_col)):
        count = data[cat_col[i]].value_counts(normalize=True).reset_index()
        sns.barplot(data=count,x='index',y=cat_col[i], ax=ax[i//7,i%7], edgecolor='white',linewidth=2,palette='Pastel1')
        ax[i//7,i%7].set_xlabel('')
        ax[i//7,i%7].set_ylabel('')
        ax[i//7,i%7].set_title(cat_col[i], size=15)
        ax[i//7,i%7].tick_params('x', labelrotation=30, size=12)
        for j in count.index:
            ax[i//7,i%7].text(x=j,y=count[cat_col[i]][j]+height, s=str(round(count[cat_col[i]][j]*100,2))+'%',ha="center", va="center", color="black",size=13)
        f.suptitle('Column Distribution', y=0.98,ha='center',va='center',size=15, weight=150)
    plt.show()

plot_categorical(raw,0)


# Majority of the features are categorical, thus, neither median or mean would be applicable. Missing values will be imputed with mode in the according feature.

# In[9]:


def impute_na(df):
    '''
    This function replaces null value in the dataset with the mode
    '''
    na_cols = [col for col in df.columns]
    for col in na_cols:
        md = df[col].mode()[0]
        df[col] = df[col].fillna(md)
    return df


# In[ ]:


# raw[raw.select_dtypes('object').columns]=raw[raw.select_dtypes('object').columns].fillna('missing')


# In[10]:


raw=impute_na(raw)
print('Dataset contains {} lines with missing value.'.format(raw.isna().sum().sum()))


# In[11]:


# display the distribution of numerical features
lst=raw.select_dtypes(['float64','int64']).columns
_,ax=plt.subplots(int(len(lst)**0.5),int(len(lst)**0.5)+1,figsize=(20,20))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
for col,axes in zip(lst,np.ravel(ax)):
  sns.boxplot(data=raw,x='Status',y=col,ax=axes)


# In[12]:


# a tabular view of the numerical features 
raw.describe()


# Extreme values in the numerical features are explainable, we will proceed with the extreme values without removing them as outliers. Now the dataset is inspected for data integrity, we may move forward to develop classification models.

# In[ ]:


sns.pairplot(raw,hue='Status')


# In[ ]:


raw.describe()


# In[ ]:





# ## Model Development

# 

# In[13]:


# convert catagorical variables to dummy variables
df=pd.get_dummies(raw,drop_first=True)


# In[ ]:


# # KNN imputor
# from sklearn.experimental import enable_iterative_imputer 
# from sklearn.impute import IterativeImputer, KNNImputer
# imp=KNNImputer(n_neighbors=2)
# X=pd.get_dummies(raw.drop('Status',axis=1),drop_first=True)
# X1=imp.fit_transform(X)


# In[14]:


# A final check before creating training/testing set
print('Number of missing values in the dataset is {}'.format(df.isna().sum().sum()))
print('Number of features in the dataset is {}'.format(len(df.columns)))


# 51 features seems overwhelming, we will inspect the correlation between the attributes and the target variable and move those with too low ($<0.05$) a correlation.

# In[ ]:


_,ax = plt.subplots(1,1,figsize=(16,12))
sns.barplot( x = df.corr().Status.drop('Status').values, y = df.corr().Status.drop('Status').index, palette='coolwarm', ax=ax)
ax.set_title('Correlation between Attributes and Target')
ax.set_xlabel('Correlation Coefficient')
plt.axvline(0.05, 0,1,linestyle='-.',c='r')
plt.axvline(-0.05,0,1,linestyle='-.',c='r')
plt.show()


# In[15]:


feats=df.corr().Status.drop('Status')[(df.corr().Status.drop('Status').apply(np.abs)>=0.05)].index
len(feats)


# In[16]:


a=pd.DataFrame(feats)
a.index=list(range(1,17))


# In[17]:


df[feats].dtypes


# In[30]:


pip install graphviz


# In[20]:


# load packages for model development
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay, make_scorer, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# In[ ]:


# X1=pd.DataFrame(X1)
# X1.columns=X.columns


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(
    # df.drop('Status',axis=1),
    # X1,
                                                    df[feats],
                                                    df['Status'],
                                                    test_size=0.3, 
                                                    random_state=42
                                                    )


# In[22]:


from sklearn.metrics.pairwise import normalize
def plot_curves(model,X_test,y_test,X_train=X_train,y_train=y_train):
  '''
  This function plot ROC curve, precision-recall curve and confusion matrix
  '''
  print("Score on the training set: {}".format(model.score(X_train,y_train)))
  print("Score on the testing set: {}".format(model.score(X_test,y_test)))
  print(classification_report(y_test,model.predict(X_test)))
  _,ax=plt.subplots(2,2,figsize=(16,16))
  font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : 22}
  plt.rc('font', **font)  
  RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax[0][0])
  PrecisionRecallDisplay.from_estimator(model,X_test,y_test,ax=ax[0][1])
  cm=confusion_matrix(y_test,model.predict(X_test))
  
  ConfusionMatrixDisplay.from_estimator(model,X_test,y_test, ax=ax[1][0],colorbar=False, cmap='YlGn')
  ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,normalize='true',ax=ax[1][1],colorbar=False, cmap='YlGn',values_format='.4f')

  ax[1][0].set_title('Confusion Matrix in Counts')
  ax[1][1].set_title('Confusion Matrix in Proportion')
  plt.show()


# Split the data into training set and testing set, keeping 30% of the data for validation.

# In[ ]:


#class_weights = dict(zip(y_train.unique(),class_weight.compute_class_weight(class_weight='balanced',classes=y_train.unique(),y=y_train)))


# In[24]:


# train a logistic regression model and plot the ROC_curve, the Precision-Recall Curve, and the Confusion Matrix in Counts and Proportion
lg_reg=LogisticRegression()
lg_reg.fit(X_train,y_train)

plot_curves(lg_reg,X_test,y_test,X_train=X_train,y_train=y_train)


# The balanced accuracy is 0.86, however, the costs are imbalanced to the financial institutes between approving loans to customers will default and denying applicants that will keep current. The logistic regression classifier misclassified $13\%$ of the actual default loans, for the purpose of this model, such a false positive rate is unacceptable and will incur tremendous loss in default loans. Considering the size of this dataset (~$150k$), we expect a decision tree classifier perform better than a logistic regression classier.

# In[25]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
# sns.reset_defaults()
# sns.set(font_scale=2)
plot_curves(dt,X_test,y_test,X_train=X_train,y_train=y_train)


# The decision tree model validated our guess, the accuracy is significantly improved with a marginal false positive rate. It only misses $20$ for every $10,000$ loan applications. There is also a considerable reduction in the mistake denials. 
# We want to rule out the possiblity that the good performance is due to a lucky-draw when splitting the data into training and testing sets so that we don't celebrate prematurely. A 10-fold cross validation will be conducted to assure the model indeed works.

# In[26]:


def fpr(y_test,y_pred):
  cm=confusion_matrix(y_test,y_pred)
  FP = cm[1][0]
  FN = cm[0][1]
  TP = cm[0][0]
  TN = cm[1][1]
  return FP/(FP+TN)

fpr_=make_scorer(fpr,greater_is_better=False)

def get_score(model,attr,target,fold=10):
  cv_fpr=cross_val_score(model, attr, target, cv=fold,scoring=fpr_)
  cv_bac=cross_val_score(model, attr, target, cv=fold, scoring='balanced_accuracy')
  print(f"Average balanced accuracy: %.4f"%(np.mean(cv_bac)))
  print(f"Standard Deviation of balanced accuracy: %.4f"%(np.std(cv_bac)),'\n')
  print(f"Average false positive rate: %.4f"%(np.mean(cv_fpr)*-1))
  print(f"Standard Deviation of False Positive Rate: %.4f"%(np.std(cv_fpr)))

get_score(dt,df[feats],df['Status'])


# In[27]:


# random forest
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
plot_curves(rfc,X_test,y_test,X_train=X_train)
# get_score(rfc,df[feats],df['Status'])


# In[28]:


xgb=XGBClassifier(random_state=42)
xgb.fit(X_train,y_train)
plot_curves(xgb,X_test, y_test, X_train=X_train,y_train=y_train)
# get_score(xgb,df[feats],df['Status'])


# In[ ]:


# import matplotlib
# plt.figure(figsize=(100,75))
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 100}
# matplotlib.rc('font', **font)
# from sklearn import tree
# _ = tree.plot_tree(dt, 
#                    feature_names=df[feats].columns,  
#                    class_names=df['Status'].unique().astype('str'),
#                    filled=True)
# plt.show()


# In[ ]:


# graph.render("decision_tree_graphivz")


# Decision Tree, Random Forest and XGBooster classifiers all achieved desirable results with a nearly zero false positive rate and an approximately $100\%$ true positive rate, namely, minimizing the two costing events in loan approvals.
# 
# We performed hyperparameter tuning on the logistic regression model to find out if the model can be more accurate with a suitable set of parameters.

# In[32]:


from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

'''
StandardScaler is used to remove the outliners and 
scale the data by making the mean of the data 0 and standard deviation as 1. 
So we are creating an object std_scl to use standardScaler.
'''
std_slc = StandardScaler()


'''
We are also using Principal Component Analysis(PCA) which will reduce the 
dimension of features by creating new features which have most of the varience of the original data.
'''
pca = decomposition.PCA()

logistic_Reg = LogisticRegression()
# decisiontree=DecisionTreeClassifier()

'''
Pipeline will helps us by passing modules one by one through GridSearchCV for which we want to get the best parameters. 
So we are making an object pipe to create a pipeline for all the three objects std_scl, pca and logistic_Reg
'''
pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('logistic_Reg', logistic_Reg)])

'''
Now we have to define the parameters that we want to optimise for these three objects.
StandardScaler doesnot requires any parameters to be optimised by GridSearchCV.
Principal Component Analysis requires a parameter 'n_components' to be optimised. 
'n_components' signifies the number of components to keep after reducing the dimension.
'''
n_components = list(range(8,X_train.shape[1]+1,1))


'''
Logistic Regression requires two parameters 'C' and 'penalty' to be optimised by GridSearchCV. 
So we have set these two parameters as a list of values form which GridSearchCV will select the best value of parameter.
'''
C = np.logspace(2, 4, 10)
penalty = ['l1', 'l2']


'''
Now we are creating a dictionary to set all the parameters options for different modules.
'''
parameters = dict(pca__n_components=n_components,
                      logistic_Reg__C=C,
                      logistic_Reg__penalty=penalty)


# parameters = dict(pca__n_components=n_components,
#                   decision_tree__criterion=criterion,
#                   )


# In[33]:


'''
Making an object clf_GS for GridSearchCV and fitting the dataset
'''
clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(X_train, y_train)
plot_curves(clf_GS, X_test,y_test,X_train=X_train, y_train=y_train)


# In[ ]:


# clf_GS = GridSearchCV(pipe, parameters)
# clf_GS.fit(X_sm,y_sm)
# plot_curves(clf_GS,X_test,y_test, X_train=X_sm, y_train=y_sm)


# In[34]:


# rfc.feature_importances_
# dir(rfc)
plt.figure(figsize=(10,3))
sns.reset_orig()
sns.barplot(x=dt.feature_names_in_,y=dt.feature_importances_)

plt.xticks(rotation=90)


# In[35]:


# plt.figure(figsize=(10,3))
sns.reset_orig()
sns.barplot(x=lg_reg.feature_names_in_,y=np.abs(lg_reg.coef_[0]),hue=lg_reg.coef_[0]/np.abs(lg_reg.coef_[0]))
# sns.barplot(x=rfc.feature_names_in_,y=rfc.feature_importances_)
plt.xticks(rotation=90)


# In[ ]:


lg_reg.coef_


# In[36]:


def logit_pvalue(model, x):
  from scipy.stats import norm
  """ Calculate z-scores for scikit-learn LogisticRegression.
  parameters:
      model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
      x:     matrix on which the model was fit
  This function uses asymtptics for maximum likelihood estimates.
  """
  p = model.predict_proba(x)
  n = len(p)
  m = len(model.coef_[0]) + 1
  coefs = np.concatenate([model.intercept_, model.coef_[0]])
  x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
  ans = np.zeros((m, m))
  for i in range(n):
      ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
  vcov = np.linalg.inv(np.matrix(ans))
  se = np.sqrt(np.diag(vcov))
  t =  coefs/se  
  p = (1 - norm.cdf(abs(t))) * 2
  return p


# In[39]:


#clf_sm=LogisticRegression()
#clf_sm.fit(X_sm,y_sm)
#plot_curves(clf_sm,X_test,y_test,X_train=X_sm,y_train=y_sm)


# In[40]:


#logit_pvalue(clf_sm,X_sm)


# In[42]:


#print(len(clf_sm.feature_names_in_))
#pd.DataFrame(pd.Series(dict(zip(clf_sm.feature_names_in_,clf_sm.coef_[0]))))


# In[43]:


#resample using over_sampling

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy="auto")
X_sm, y_sm = smote.fit_resample(X_train, y_train)


# In[44]:


pd.Series(y_sm.value_counts()).plot(kind='pie',
                                                                        title='Proportions of Loan Status',
                                                                        #  explode=(0,0.1),
                                                                        autopct=lambda p:f'{p:.2f}%, {int(p*len(df)/100)}',
                                                                        labels=['',''],
                                                                        ylabel='',
                                                                        cmap='coolwarm'
                                                                        )
font = {'family' : 'DejaVu Sans',
'weight' : 'bold',
'size'   : 15}
plt.rc('font', **font)  
plt.style.use('fast')
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.legend(['Paid-off (0)','Default (1)'])
plt.show()


# In[49]:


pca=PCA(n_components=18)

X_pca_train=pca.fit_transform(X_train)
X_pca_test=pca.transform(X_test)


# In[47]:


xgb2=XGBClassifier()
xgb2.fit(X_pca_train,y_train)
plot_curves(xgb2,X_pca_test,y_test,X_train=X_pca_train)


# In[48]:


lgr2=LogisticRegression()
clf_GS.fit(X_sm,y_sm)
plot_curves(clf_GS,X_test,y_test,X_train=X_sm,y_train=y_sm)


# In[50]:


std_slc = StandardScaler()
pca = decomposition.PCA()
logistic_Reg = LogisticRegression()

pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('logistic_Reg', logistic_Reg)])
n_components = list(range(8,X_train.shape[1]+1,1))
parameters = dict(pca__n_components=n_components,)


lgcv=GridSearchCV(pipe,parameters,scoring=fpr_)


# In[51]:


lgcv.fit(X_train,y_train)


# In[52]:


# with full features (PCA down select to 47 from 54)
plot_curves(lgcv,X_test,y_test,X_train=X_train)
lgcv.best_params_


# In[53]:


# with pre-selected features based on correlation with the target variable (PCA down select to 14 from 19)
plot_curves(lgcv,X_test,y_test,X_train=X_train)
lgcv.best_params_


# In[55]:


ft.columns=['Attribute']
ft


# In[56]:


raw.loan_amount.median()


# In[57]:


raw.property_value.median()


# In[ ]:





# In[58]:


pd.Series(np.round(clf_GS.predict_proba(X_test))[:,1]).value_counts()


# In[ ]:


y_test.value_counts()


# In[ ]:





# In[ ]:


coef=pd.DataFrame([lg_reg.feature_names_in_, lg_reg.coef_[0]]).T
coef.index=range(1,17)
coef


# 

# In[59]:


mmscaler=MinMaxScaler()


# In[60]:


Xtn_mm=mmscaler.fit_transform(X_train)
Xtt_mm=mmscaler.transform(X_test)


# In[61]:


lg_reg.fit(X_train,y_train)


# In[62]:


lg_reg.coef_


# In[64]:


#p=logit_pvalue(lg_reg,X_train)


# In[65]:


p[1:]


# In[ ]:


pd.DataFrame([lg_reg.feature_names_in_,p[1:]]).T


# In[ ]:


y_test[3837]


# In[ ]:





# In[66]:


xt=X_test.iloc[24912]


# In[67]:


xt['Interest_rate_spread']=-5


# In[68]:


lg_reg.predict_proba(np.array(xt).reshape(1,-1))


# In[69]:


y_test.iloc


# In[70]:


raw['submission_of_application'].unique()


# In[ ]:




