#!/usr/bin/env python
# coding: utf-8

# In this study, I examined dataset with visualization techniques and I applied Machine Learning algorithms to the dataset. Also, I created DecisionTreeClassifier model. 
# 

# Importing Libraries

# In[2]:


import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn import tree
import graphviz 
from sklearn.model_selection import GridSearchCV


# # Information About Data

# In this part, you can see details about dataset.

# In[3]:


train_set = pandas.read_csv('/kaggle/input/mobile-price-classification/train.csv')
print(train_set)


# In[4]:


print(train_set.head())


# In[5]:


train_set.info()


# In[6]:


train_set.describe()


# In[7]:


print(train_set.describe(include='all'))


# # Pandas Profiling

# Pandas profiling is a useful library that generates interactive reports about the data. With using this library, we can see types of data, distribution of data and various statistical information. This tool has many features for data preparing. Pandas Profiling includes graphics about specific feature and correlation maps too. You can see more details about this tool in the following url: https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/

# In[8]:


from pandas_profiling import ProfileReport
prof = ProfileReport(train_set)
prof.to_file(output_file='report.html')


# # Data Visualization

# In[9]:


fig, ax = plt.subplots()
ax.scatter(train_set['talk_time'], train_set['price_range'])
ax.set_title('Mobile Phones Dataset')
ax.set_xlabel('battery_power')
ax.set_ylabel('price_range')


# In[10]:


fig, ax = plt.subplots()
ax.hist(train_set['battery_power'])
ax.set_title('Battery Power Scores')
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')


# In[11]:


sns.jointplot(x='fc',y='price_range',data=train_set,color='green',kind='kde');


# In[12]:


sns.boxplot(x="price_range", y="ram", data=train_set)


# In[13]:


sns.boxplot(x="price_range", y="n_cores", data=train_set)


# In[14]:


sns.distplot(train_set['px_height'])


# In[15]:


sns.distplot(train_set['px_width'])


# In[16]:


plt.figure(figsize=(10,6))
sns.barplot(x=train_set.pc, y=train_set['price_range'])


# # Detecting Correlations

# In[17]:



features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 
            'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 
            'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 
            'three_g', 'touch_screen', 'wifi', 'price_range']

mask = np.zeros_like(train_set[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(train_set[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});


# In[18]:


sns.heatmap(train_set[features].corr())


# According to these correlation maps, there are positive correlations among some features. For example, when px_height increase, px_width increase too. Also, these features have positive correlation with price_range. We can say that these features are affected product's price. fc and pc have positive correlation too. So, it is possible to say that, when one them value increase, other's is increase too.

# In[19]:


def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 200
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
data = train_set
columns = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 
            'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 
            'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 
            'three_g', 'touch_screen', 'wifi', 'price_range'] 
corr = data[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# # Data Preprocessing

# In[20]:


feature = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 
            'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 
            'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 
            'three_g', 'touch_screen', 'wifi', 'price_range']

label = 'price_range'

X = train_set.drop(['price_range'], axis=1) 
y = train_set[label]


# In[21]:


print(X[:2])


# Normalizing Dataset

# In[22]:


X = normalize(X, norm='l2')
print(X[:3])


# # Splitting Dataset 

# Splitting dataset into train, test and validation.

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# In[24]:


print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in validation dataset: {len(X_valid)}')
print(f'Total # of sample in test dataset: {len(X_test)}')


# # Scores of Models

# In[25]:


models = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB(),
    'LinearRegression()': LinearRegression(),
    'LogisticRegression': LogisticRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'CategoricalNB': CategoricalNB(),
    'KNN': KNeighborsClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

for m in models:
  model = models[m]
  model.fit(X_train, y_train)
  score = model.score(X_valid, y_valid)
  #print(f'{m} validation score => {score*100}')
    
  print(f'{m}') 
  train_score = model.score(X_train, y_train)
  print(f'Train score of trained model: {train_score*100}')

  validation_score = model.score(X_valid, y_valid)
  print(f'Validation score of trained model: {validation_score*100}')

  test_score = model.score(X_test, y_test)
  print(f'Test score of trained model: {test_score*100}')
  print(" ")


# This code includes scores of some Machine Learning models. According to these results, DecisionTreeClassifier and RandomForestRegressor will be useful for this dataset.

# # DecisionTreeClassifier

# Creating DecisionTreeClassifier model.

# In[26]:


dtree_model = DecisionTreeClassifier()
dtree_model.fit(X_train, y_train)

train_score = dtree_model.score(X_train, y_train)
print(f'Train score of trained model: {train_score*100}')

validation_score = dtree_model.score(X_valid, y_valid)
print(f'Validation score of trained model: {validation_score*100}')

test_score = dtree_model.score(X_test, y_test)
print(f'Test score of trained model: {test_score*100}')


# In[27]:


y_predictions = dtree_model.predict(X_test)

conf_matrix = confusion_matrix(y_predictions, y_test)
print("Confusion Matrix")
print(conf_matrix)
print("\n")
print(f'Accuracy: {accuracy_score(y_predictions, y_test)*100}')


# In[28]:


tn = conf_matrix[0,0]
fp = conf_matrix[0,1]
tp = conf_matrix[1,1]
fn = conf_matrix[1,0]

accuracy  = (tp + tn) / (tp + fp + tn + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1score  = 2 * precision * recall / (precision + recall)

print(f'Accuracy : {accuracy*100}')
print(f'Precision: {precision*100}')
print(f'Recall   : {recall*100}')
print(f'F1 score : {f1score*100}')


# In[29]:


sns.heatmap(conf_matrix, annot=True)


# In[30]:


print(f'Test score of trained DecisionTreeClassifier model: {dtree_model.score(X_test, y_test)*100}')


# In[31]:


print(classification_report(y_predictions, y_test))


# # Visualization of DecisionTreeClassifier

# In[32]:


plt.figure(figsize=(10,8))
tree.plot_tree(dtree_model) 
plt.show()


# In[33]:


exported_tree = tree.export_graphviz(dtree_model) 
tree_plot = graphviz.Source(exported_tree)
tree_plot


# To save as .pdf file
# 
# tree_plot.render('decision_tree')

# In[34]:


exported_tree = tree.export_graphviz(dtree_model, 
                                     feature_names = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 
                                                      'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 
                                                      'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 
                                                      'three_g', 'touch_screen', 'wifi'],  
                                     class_names = label,  
                                     filled = True, rounded = True,  
                                     special_characters = True)

tree_plot = graphviz.Source(exported_tree)  
tree_plot


# To save as .pdf file
# 
# tree_plot.render('color_tree2_dt')

# In[35]:


dtree_model.predict(X_test)


# # Hyper-parameter Tuning

# In[36]:


gcv = GridSearchCV(DecisionTreeClassifier(), {'max_depth': range(1,30), 
                                             'min_samples_split': range(2,8), 
                                             'min_samples_leaf': range(2, 8)}).fit(X_train, y_train)


# In[37]:


pandas.DataFrame(gcv.cv_results_)


# In[38]:


gcv.best_estimator_


# In[39]:


gcv.best_params_


# In[40]:


gcv.best_score_*100


# In[41]:


print(f'Initial model: {dtree_model.score(X_test, y_test)*100}')
print(f'Optimal model: {gcv.score(X_test, y_test)*100}')

