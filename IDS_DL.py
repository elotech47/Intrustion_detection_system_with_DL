#!/usr/bin/env python
# coding: utf-8

#  # Intrusion Detection System with deep learning

# In[2]:


#Import neccessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


monday_data = pd.read_csv("/content/drive/MyDrive/engEdosa/Dataset/Monday-WorkingHours.pcap_ISCX.csv")
tuesday_data = pd.read_csv("/content/drive/MyDrive/engEdosa/Dataset/Tuesday-WorkingHours.pcap_ISCX.csv")
wednesday_data = pd.read_csv("/content/drive/MyDrive/engEdosa/Dataset/Wednesday-workingHours.pcap_ISCX.csv")
thursday_data_1 = pd.read_csv("/content/drive/MyDrive/engEdosa/Dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
thursday_data_2 = pd.read_csv("/content/drive/MyDrive/engEdosa/Dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
friday_data_1 = pd.read_csv("/content/drive/MyDrive/engEdosa/Dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
friday_data_2 = pd.read_csv("/content/drive/MyDrive/engEdosa/Dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
friday_data_3 = pd.read_csv("/content/drive/MyDrive/engEdosa/Dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv")


# In[ ]:


# wednesday_data.columns


# In[ ]:


thursday_data_1.shape


# In[ ]:


data_details = {
    "monday_details":monday_data[' Label'].value_counts(),
     "tuesday_details":tuesday_data[' Label'].value_counts(),
      "wednesday_details":wednesday_data[' Label'].value_counts(),
       "thursday_details_1":thursday_data_1[' Label'].value_counts(),
       "thursday_details_2":thursday_data_2[' Label'].value_counts(),
       "friday_details_1":friday_data_1[' Label'].value_counts(),
       "friday_details_2":friday_data_2[' Label'].value_counts(),
       "friday_details_3":friday_data_3[' Label'].value_counts()
}


# In[ ]:


data_details


# In[ ]:


frames = [wednesday_data, friday_data_1, friday_data_2]


# In[ ]:


data = pd.concat(frames)


# In[ ]:


data.shape


# In[ ]:


#data.describe()


# In[ ]:


data[' Label'].value_counts()


# In[ ]:


#data.sample(10)


# In[ ]:


# # Getting a sense of what the distribution of each column looks like
# fig = plt.figure(figsize=(15,10))

# ax1 = fig.add_subplot(221)
# data[' Label'].value_counts().plot(kind='bar', ax=ax1)
# ax1.set_ylabel('Count')
# ax1.set_title('Label');

# plt.tight_layout()
# plt.show()


# In[ ]:


# data.isna().sum()


# In[ ]:


np.isinf(data[" Flow Duration"]).sum()


# In[ ]:


max_flow_bytes = data.loc[data['Flow Bytes/s'] != np.inf, 'Flow Bytes/s'].max()
max_flow_pkts = data.loc[data[' Flow Packets/s'] != np.inf, ' Flow Packets/s'].max()

print(max_flow_bytes, max_flow_pkts)


# In[ ]:





# In[ ]:


data['Flow Bytes/s'].replace(np.inf,max_flow_bytes+1,inplace=True)
data[' Flow Packets/s'].replace(np.inf,max_flow_pkts+1,inplace=True)


# In[ ]:


data[' Label'].value_counts()
#data[['Date','Time']] = data['Timestamp'].str.split(expand=True)


# In[ ]:





# In[ ]:


Mal = {'BENIGN':0, 'FTP-Patator':1, 'SSH-Patator':1, 'DoS slowloris':1,
       'DoS Slowhttptest':1, 'DoS Hulk':1, 'DoS GoldenEye':1, 'Heartbleed':1,
       'Web Attack � Brute Force':1, 'Web Attack � XSS':1,
       'Web Attack � Sql Injection':1, 'Infiltration':1, 'DDoS':1, 'PortScan':1,
       'Bot':1}
data[' Label'] = [Mal[item] for item in data[' Label']]


# In[ ]:



# # Getting a sense of what the distribution of each column looks like
# fig = plt.figure(figsize=(15,10))

# ax1 = fig.add_subplot(221)
# data[' Label'].value_counts().plot(kind='bar', ax=ax1)
# ax1.set_ylabel('Count')
# ax1.set_title('Label');

# plt.tight_layout()
# plt.show()


# In[ ]:


data.shape


# In[ ]:


data.columns = data.columns.str.strip()
df = data.drop(columns=["Fwd Header Length.1"])
df.shape


# In[ ]:


df['Label'].value_counts()


# In[ ]:


df.replace('Infinity', -1, inplace=True)
df[["Flow Bytes/s", "Flow Packets/s"]] = df[["Flow Bytes/s", "Flow Packets/s"]].apply(pd.to_numeric)


# In[ ]:


df.replace([np.nan], -1, inplace=True)


# In[ ]:


# df.describe()


# In[ ]:


# df.to_csv("/content/drive/MyDrive/engEdosa/Dataset/web_attacks_unbalanced.csv", index=False)
# df['Label'].value_counts()


# In[ ]:


benign_total = len(df[df['Label'] == 0])
attack_total = len(df[df['Label'] != 0])
attack_total


# In[ ]:


df.tail()


# In[ ]:


df.to_csv("/content/drive/MyDrive/engEdosa/Dataset/web_attacks_balanced.csv", index=False)


# In[4]:


df = pd.read_csv("/content/drive/MyDrive/engEdosa/Dataset/web_attacks_balanced.csv")


# 7 features (Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp) are excluded from the dataset. The hypothesis is that the "shape" of the data being transmitted is more important than these attributes. In addition, ports and addresses can be substituted by an attacker, so it is better that the ML algorithm does not take these features into account in training [Kostas2018].

# In[5]:


excluded = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
df = df.drop(columns=excluded, errors='ignore')


# In[ ]:


df.columns


# Below at the stage of importance estimation the "Init_Win_bytes_backward" feature has the maximum value. After viewing the source dataset, it seems that an inaccuracy was made in forming the dataset. 
# 
# It turns out that it is possible to make a fairly accurate classification by one feature.
# 
# Description of features: http://www.netflowmeter.ca/netflowmeter.html
# 
#      Init_Win_bytes_backward - The total number of bytes sent in initial window in the backward direction
#      Init_Win_bytes_forward - The total number of bytes sent in initial window in the forward direction

# In[ ]:


if 'Init_Win_bytes_backward' in df.columns:
    df['Init_Win_bytes_backward'].hist(figsize=(6,4), bins=10);
    plt.title("Init_Win_bytes_backward")
    plt.xlabel("Value bins")
    plt.ylabel("Density")
    plt.savefig('Init_Win_bytes_backward.png', dpi=300)


# In[ ]:


if 'Init_Win_bytes_forward' in df.columns:
    df['Init_Win_bytes_forward'].hist(figsize=(6,4), bins=10);
    plt.title("Init_Win_bytes_forward")
    plt.xlabel("Value bins")
    plt.ylabel("Density")
    plt.savefig('Init_Win_bytes_forward.png', dpi=300)


# In[6]:


excluded2 = ['Init_Win_bytes_backward', 'Init_Win_bytes_forward']
df = df.drop(columns=excluded2, errors='ignore')


# In[7]:


y = df['Label'].values
X = df.drop(columns=['Label'])
print(X.shape, y.shape)


# ## Feature importance

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))


# ### Visualization of the decision tree, importance evaluation using a single tree (DecisionTreeClassifier)
#  

# In[9]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
decision_tree = decision_tree.fit(X_train, y_train)
cross_val_score(decision_tree, X_train, y_train, cv=10)


# In[10]:


from sklearn.tree import export_text
r = export_text(decision_tree, feature_names=X_train.columns.to_list())
print(r)


# In[11]:


from graphviz import Source
from sklearn import tree
Source(tree.export_graphviz(decision_tree, out_file=None, feature_names=X.columns))


# Analyze the confusion matrix. Which classes are confidently classified by the model?

# In[ ]:


unique, counts = np.unique(y_test, return_counts=True)
dict(zip(unique, counts))


# In[ ]:


from sklearn.metrics import confusion_matrix
y_pred = decision_tree.predict(X_test)
confusion_matrix(y_test, y_pred)


# ### Importance evaluation using SelectFromModel (still one decision tree)
# 

# In[ ]:


from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(estimator=decision_tree).fit(X_train, y_train)
sfm.estimator_.feature_importances_


# In[ ]:


sfm.threshold_


# In[ ]:


X_train_new = sfm.transform(X_train)
print("Original num features: {}, selected num features: {}"
      .format(X_train.shape[1], X_train_new.shape[1]))


# In[ ]:


indices = np.argsort(decision_tree.feature_importances_)[::-1]
for idx, i in enumerate(indices[:10]):
    print("{}.\t{} - {}".format(idx, X_train.columns[i], decision_tree.feature_importances_[i]))


# ### Evaluation of importance using RandomForestClassifier.feature_importances_ (move from one tree to a random forest, classification quality increases)

# In[13]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=250, random_state=42, oob_score=True)
rf.fit(X_train, y_train)
# Score = mean accuracy on the given test data and labels
print('R^2 Training Score: {:.2f} \nR^2 Validation Score: {:.2f} \nOut-of-bag Score: {:.2f}'
      .format(rf.score(X_train, y_train), rf.score(X_test, y_test), rf.oob_score_))


# In[16]:


features = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
webattack_features = []


for index, i in enumerate(indices[:20]):
    webattack_features.append(features[i])
    print('{}.\t#{}\t{:.3f}\t{}'.format(index + 1, i, importances[i], features[i]))


# In[21]:


indices


# In[27]:


import pandas as pd
forest_importances = pd.Series(importances[indices[0:20]], webattack_features)

fig, ax = plt.subplots()
forest_importances.plot.bar( ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Importance")
fig.tight_layout()


# 
# 

# In[28]:


indices = np.argsort(importances)[-20:]
plt.rcParams['figure.figsize'] = (10, 6)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='#cccccc', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.grid()
plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()


# In[29]:


y_pred = rf.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[ ]:


max_features = 20
webattack_features = webattack_features[:max_features]
webattack_features


# ## Analysis of selected features

# In[ ]:


df[webattack_features].hist(figsize=(20,12), bins=10);
plt.savefig('features_hist.png', dpi=300)


# In[ ]:


get_ipython().system('pip install facets-overview')


# In[ ]:


import base64
from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator

gfsg = GenericFeatureStatisticsGenerator()
proto = gfsg.ProtoFromDataFrames([{'name': 'train + test', 'table': df[webattack_features]}])
protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")


# In[ ]:


from IPython.core.display import display, HTML

HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
        <facets-overview id="elem"></facets-overview>
        <script>
          document.querySelector("#elem").protoInput = "{protostr}";
        </script>"""
html = HTML_TEMPLATE.format(protostr=protostr)
display(HTML(html))


# In[ ]:


import seaborn as sns
corr_matrix = df[webattack_features].corr()
plt.rcParams['figure.figsize'] = (16, 5)
g = sns.heatmap(corr_matrix, annot=True, fmt='.1g', cmap='Greys')
g.set_xticklabels(g.get_xticklabels(), verticalalignment='top', horizontalalignment='right', rotation=30);
plt.savefig('/content/drive/MyDrive/engEdosa/corr_heatmap.png', dpi=300, bbox_inches='tight')


# In[ ]:


to_be_removed = {'Packet Length Mean', 'Avg Fwd Segment Size', 'Subflow Fwd Bytes', 
                 'Fwd Packets/s', 'Fwd IAT Total', 'Fwd IAT Max'}
webattack_features = [item for item in webattack_features if item not in to_be_removed]
webattack_features = webattack_features[:10]
webattack_features


# In[ ]:


corr_matrix = df[webattack_features].corr()
plt.rcParams['figure.figsize'] = (6, 5)
sns.heatmap(corr_matrix, annot=True, fmt='.1g', cmap='Greys');


# ## Model Training

# In[30]:


y = df['Label'].values
X = df[webattack_features]
print(X.shape, y.shape)


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[34]:


import pyforest
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[36]:


import lazypredict
from lazypredict.Supervised import LazyClassifier


# In[1]:


clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric = None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# In[ ]:




