#!/usr/bin/env python
# coding: utf-8

# # 1. import files

# In[1]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

#-- load in the data
filename = r"C:\Users\15431\Desktop\ASA\course work\category_spends_sample_3.csv"
data = pd.read_csv(filename)
bs=pd.read_csv(r"C:\Users\15431\Desktop\ASA\course work\baskets_sample.csv")
cs=pd.read_csv(r"C:\Users\15431\Desktop\ASA\course work\customers_sample_3.csv")
ls=pd.read_csv(r"C:\Users\15431\Desktop\ASA\course work\lineitems_sample.csv")


# # 2.clean the data

# In[2]:


#clean the dirty data
ls_clean=ls.copy()
ls_clean=ls_clean[ls_clean["quantity"]>=1]

# purchase_time 是时间变量，需要进行调整；
#ls_clean["purchase_month"]=ls_clean["purchase_time"].astype("datetime64[M]")
ls_clean["purchase_time"]=ls_clean["purchase_time"].astype("datetime64[M]")

# basket_spend 将object 类型转换为浮点型数据；将英镑符号数据转换成可以进行计算的数据
# https://www.codeleading.com/article/65724021889/
def clean_currency(x):
    if isinstance(x, str):
        return(x.replace('£', '').replace(',', ''))
    return(x)

ls_clean['spend']= ls_clean['spend'].apply(clean_currency).astype('float')

ls_clean.head()
ls_clean.dtypes

#删除basket_spend的异常数据
bs_clean=bs.copy()
bs_clean[bs_clean['basket_spend'].isin(["#NAME?"])]
bs_clean=bs_clean[~bs_clean['basket_spend'].isin(["#NAME?"])]
bs_clean

#删除basket_quantity的异常数据
bs_clean[bs_clean['basket_quantity']<=0]
bs_clean=bs_clean[~bs_clean['basket_quantity']<=0]
bs_clean

#更改basket_spend的数据类型
def clean_currency(x):
    if isinstance(x, str):
        return(x.replace('£', '').replace(',', ''))
    return(x)

bs_clean['basket_spend']= bs_clean['basket_spend'].apply(clean_currency).astype('float')

bs_clean.head()

# purchase_time 是时间变量，需要进行调整；
bs_clean["test"] = pd.to_datetime(bs_clean["purchase_time"],format='%Y-%m-%d %H:%M:%S')#将读取的日期转为datatime格式
bs_clean['new_time'] = bs_clean['test'].dt.strftime("%H:%M")
bs_clean["final"] = pd.to_datetime(bs_clean["new_time"],format='%H:%M')

bs_clean.dtypes


# # 3.extract features in bs_clean

# In[3]:


bs_clean


# In[4]:


a=pd.to_datetime('1900-01-01 06:00:00',format='%Y-%m-%d %H:%M:%S')
b=pd.to_datetime('1900-01-01 07:00:00',format='%Y-%m-%d %H:%M:%S')
c=pd.to_datetime('1900-01-01 08:00:00',format='%Y-%m-%d %H:%M:%S')
d=pd.to_datetime('1900-01-01 09:00:00',format='%Y-%m-%d %H:%M:%S')
e=pd.to_datetime('1900-01-01 10:00:00',format='%Y-%m-%d %H:%M:%S')
f=pd.to_datetime('1900-01-01 11:00:00',format='%Y-%m-%d %H:%M:%S')
g=pd.to_datetime('1900-01-01 12:00:00',format='%Y-%m-%d %H:%M:%S')
h=pd.to_datetime('1900-01-01 13:00:00',format='%Y-%m-%d %H:%M:%S')
i=pd.to_datetime('1900-01-01 14:00:00',format='%Y-%m-%d %H:%M:%S')
j=pd.to_datetime('1900-01-01 15:00:00',format='%Y-%m-%d %H:%M:%S')
k=pd.to_datetime('1900-01-01 16:00:00',format='%Y-%m-%d %H:%M:%S')
l=pd.to_datetime('1900-01-01 17:00:00',format='%Y-%m-%d %H:%M:%S')
m=pd.to_datetime('1900-01-01 18:00:00',format='%Y-%m-%d %H:%M:%S')
n=pd.to_datetime('1900-01-01 19:00:00',format='%Y-%m-%d %H:%M:%S')
o=pd.to_datetime('1900-01-01 20:00:00',format='%Y-%m-%d %H:%M:%S')
p=pd.to_datetime('1900-01-01 21:00:00',format='%Y-%m-%d %H:%M:%S')
q=pd.to_datetime('1900-01-01 22:00:00',format='%Y-%m-%d %H:%M:%S')
r=pd.to_datetime('1900-01-01 23:00:00',format='%Y-%m-%d %H:%M:%S')
s=pd.to_datetime('1900-01-01 23:59:59',format='%Y-%m-%d %H:%M:%S')
x=bs_clean.final

# 6 represents "06:00am-07:00am"
# 7 represents "07:00am-08:00am"
# 8 represents "08:00am-09:00am"
# 9 represents "09:00am-10:00am"
#......

bs_clean["purchase_zone"]=np.where(x<b,6,np.where(x<c,7,np.where(x<d,8,np.where(x<e,9,np.where(x<f,10,np.where(x<j,11,np.where(x<h,12,np.where(x<i,13,np.where(x<g,14,np.where(x<k,15,np.where(x<l,16,np.where(x<m,17,np.where(x<n,18,np.where(x<o,19,np.where(x<p,20,np.where(x<q,21,np.where(x<r,22,np.where(x<s,23,24))))))))))))))))))
bs_clean.head()


# In[5]:


#顾客在4个时间区间的购买次数
c=bs_clean.groupby(by=['customer_number',"purchase_zone"])['purchase_time']
c=c.agg([('purchase_time',"count")])
c.head(20)


# In[6]:


d=c.sort_values(["purchase_time"],ascending=False).groupby("customer_number").head(1)
d


# In[7]:


d=pd.DataFrame(d)
d.to_csv('d')
d=pd.read_csv('d')
d


# In[8]:


d.rename(columns={"purchase_zone":"purchase_max_frequency_zone"},inplace=True)
d


# In[9]:


d1=d.copy()
d1.drop('purchase_time',axis = 1,inplace = True)
d1


# In[10]:


bs_final=d1.copy()
bs_final


# In[11]:


display(bs_final.groupby('purchase_max_frequency_zone').agg('count'))


# # 4.extract features in data

# In[12]:


data


# In[13]:


#-- first remove the column 'customer number' 
data_clean=data.copy()
data_clean.drop(['customer_number'], axis = 1, inplace = True)

#-- detail the number of datapoints and featuers
print("Number of datapoints:", data_clean.shape[0])
print("Number of features:", data_clean.shape[1])

#-- print out some summary statistics as per normal
data_clean.describe()


# In[14]:


# get a sense of them visually
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#-- Produce a scatter matrix for each pair of features in the data
scatter = pd.plotting.scatter_matrix(data_clean, figsize = (20,10))


# In[15]:


#-- create and print a cross correlation of all the variables against each other
corr = data_clean.corr()
print(corr)


# In[16]:


#-- Plot the results using a heatmap
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.matshow(corr, cmap="Oranges")

#-- set the names of each column on the graph
plt.xticks(range(20), data_clean.columns);
plt.yticks(range(20), data_clean.columns);


# In[17]:


from sklearn.decomposition import PCA

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=4)
pca.fit(data_clean)

#-- import a helpful set of functions to ease displaying results..
import renders as rs

#-- Generate a PCA results plot 
pca_results = rs.pca_results(data_clean, pca)


# In[18]:


def display_factors(model, original_features):
   dimensions = range(1, len(model.components_) + 1)
   topics = pd.DataFrame(model.components_, columns = original_features)
   fig, ax = plt.subplots(figsize = (14,8))
   topics.plot(ax = ax, kind = 'bar');
   ax.set_ylabel("Original Feature Weights")
   ax.set_xlabel("Derived Factors")
   ax.set_xticklabels(dimensions, rotation=0)

from sklearn.decomposition import NMF

#-- Generate a PCA factorization of your data
nmf = NMF(n_components=4, random_state=1)
nmf.fit(data_clean)

#-- And visualize the results
nmf_results = display_factors(nmf, data_clean.columns)


# In[19]:


nmf.components_


# In[20]:


for t, topic in enumerate(nmf.components_):
    print("\nTOPIC", t)
    print("----------")
    
    #-- attach the feature name to each topic weighting
    weightings = list(zip(topic, data_clean.columns))
 
    #-- sort the weightings into an order
    ordered_indeces = topic.argsort()
    
    #-- make the order highest first
    reversed_indeces = ordered_indeces[::-1]
    
    #-- reduce it down to only the top 4 items
    top_3_indeces = reversed_indeces[:4]
    
    #-- print the results out to screen
    for i in top_3_indeces:
        print("{:.2f} {}".format(weightings[i][0], weightings[i][1]))


# In[21]:


feature_names = ["HOUSEWIFE", "SMOKER","DRINKER","SWEET_LOVER"]
new_features = nmf.transform(data_clean)

print(feature_names)
print(new_features)


# In[22]:


a=pd.DataFrame(new_features, columns=["HOUSEWIFE", "SMOKER","DRINKER","SWEET_LOVER"])
a


# In[23]:


#scatter = pd.plotting.scatter_matrix(a, figsize = (20,10))


# In[24]:


customer_number=data[["customer_number"]]
customer_number


# In[25]:


data_final = pd.merge(customer_number,a,left_index=True,right_index=True,how="outer")
data_final


# # 5.extract features in cs_clean

# In[26]:


cs_clean=cs.copy()
cs_clean


# In[27]:


cs_clean.drop(['customer_number'], axis = 1, inplace = True)
cs_clean


# In[28]:


#-- Produce a scatter matrix for each pair of features in the data
scatter = pd.plotting.scatter_matrix(cs_clean, figsize = (20,10))


# In[29]:


#-- create and print a cross correlation of all the variables against each other
corr = cs_clean.corr()
print(corr)


# In[30]:


#-- Plot the results using a heatmap
plt.rcParams['figure.figsize'] = (8.0, 8.0)
plt.matshow(corr, cmap="Oranges")

#-- set the names of each column on the graph
plt.xticks(range(5), cs_clean.columns);
plt.yticks(range(5), cs_clean.columns);


# In[31]:


# highly co-related, therefore just leave two feature:average_quantity and total_spend


# In[32]:


cs_final=cs.copy()
cs_final.drop(['baskets','total_quantity','average_spend'], axis = 1, inplace = True)
cs_final


# # 6.final table

# In[33]:


final_1=pd.merge(cs_final,bs_final,on="customer_number",how="outer")
final_1


# In[34]:


final_2=pd.merge(final_1,data_final,on="customer_number",how="outer")
final_2


# # 7.customer base exploration and final table dimension reduction

# In[35]:


data_clean


# In[36]:


category_sum=np.sum(data_clean, axis=0)
category_sum=pd.DataFrame(category_sum)
category_sum.columns=["total_spend"]
category_sum=category_sum.sort_values(by="total_spend", ascending=False)
category_sum


# In[37]:


category_sum.plot.bar()
plt.xlabel("category_name")
plt.title("category_spend_distribution")
plt.rcParams["figure.figsize"]=(10.0, 5.0)


# In[38]:


ls_clean


# In[39]:


category_quantity=ls_clean.groupby(by=['category'])['quantity']
category_quantity=category_quantity.agg([('quantity',"sum")])
category_quantity=category_quantity.sort_values(by="quantity", ascending=False)
category_quantity


# In[40]:


category_quantity.plot.bar()
plt.xlabel("category_name")
plt.title("category_quantity_distribution")
plt.rcParams["figure.figsize"]=(10.0, 3.0)


# In[41]:


final_reduced=final_2
final_reduced.drop(['customer_number'], axis = 1, inplace = True)
final_reduced


# In[42]:


print("Number of datapoints:", final_reduced.shape[0])
print("Number of features:", final_reduced.shape[1])


# In[43]:


final_reduced.describe()


# In[44]:


scatter = pd.plotting.scatter_matrix(final_reduced, figsize = (30,20))


# In[45]:


#-- create and print a cross correlation of all the variables against each other
corr = final_reduced.corr()
print(corr)


# In[46]:


#-- Plot the results using a heatmap
plt.rcParams['figure.figsize'] = (13.0, 13.0)
plt.matshow(corr, cmap="Oranges")

#-- set the names of each column on the graph
plt.xticks(range(7), final_reduced.columns);
plt.yticks(range(7), final_reduced.columns);


# In[47]:


import numpy as np

# Scale the data using the natural logarithm
logged_data = np.log1p(final_reduced)

#-- Produce a scatter matrix using the logged data...
scatter = pd.plotting.scatter_matrix(logged_data, figsize = (20,10))


# In[48]:


from sklearn.decomposition import PCA

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=6)
pca.fit(logged_data)

#-- import a helpful set of functions to ease displaying results..
import renders as rs

#-- Generate a PCA results plot 
pca_results = rs.pca_results(logged_data, pca)


# In[49]:


# TODO: Apply PCA by fitting the good data with only two dimensions
# Instantiate
pca = PCA(n_components=2)
pca.fit(logged_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(logged_data)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data)
scatter = pd.plotting.scatter_matrix(reduced_data, figsize = (20,10))


# # 8.clustering

# In[50]:


#-- New  imports we will need
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#-- Create a clusterer that fits to 3 segments
k = 3
clusterer = KMeans(n_clusters=k)
clusterer.fit(reduced_data)

#-- TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)
    
#-- Calculate a silhouette score for the 3 segment solution
score = silhouette_score(final_reduced, preds, metric='euclidean')
print("For n_clusters = {}. The average silhouette_score is : {})".format(k, score))


# In[51]:


# Create range of clusters 
range_n_clusters = list(range(2,11))
print(range_n_clusters)
range_score = []
# Loop through clusters
for n_clusters in range_n_clusters:
    # TODO: Apply your clustering algorithm of choice to the reduced data 
    clusterer = KMeans(n_clusters=n_clusters).fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.cluster_centers_

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds, metric='euclidean')
    range_score.append(score)
    print("For n_clusters = {}. The average silhouette_score is : {})".format(n_clusters, score))
    


# In[52]:


plt.plot(range_n_clusters, range_score)
plt.show()


# In[53]:


#-- Our final clustering solution
clusterer = KMeans(n_clusters=4).fit(reduced_data)
preds = clusterer.predict(reduced_data)
centres = clusterer.cluster_centers_

#-- Put the predictions into a pandas dataframe format
assignments = pd.DataFrame(preds, columns = ['Cluster'])

#-- Put the predictions into a pandas dataframe format
plot_data = pd.concat([assignments, reduced_data], axis = 1)

#-- Color the points based on assigned cluster (n.b scatter will do this for us automatically)
plt.rcParams['figure.figsize'] = (14.0, 8.0)

for i, c in plot_data.groupby('Cluster'):  
    plt.scatter(c[0], c[1])
    
#-- Plot where the cluster centers are
for i, c in enumerate(centres):
    plt.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', marker = 'o', s=300);
    plt.scatter(x = c[0], y = c[1], marker='${}$'.format(i), alpha = 1, s=50);


# In[54]:


# TODO: Inverse transform the centres
log_centres = pca.inverse_transform(centres)

# TODO: Exponentiate the centres
true_centres = np.exp(log_centres)

#-- Display the true centres
segments = ['Segment {}'.format(i) for i in range(0, len(centres))]
true_centres = pd.DataFrame(np.round(true_centres), columns = final_reduced.columns)
true_centres.index = segments
print(true_centres)


# In[55]:


#-- Join the segment assignments to the original data 
final_assignments = pd.concat([assignments, final_reduced], axis = 1)

#-- Create a loop that describes summary statistics for each segment
for c, d in final_assignments.groupby('Cluster'):  
    print("SEGMENT", c)
    display(d.describe())


# In[56]:


display(final_assignments.groupby('Cluster').agg('mean'))


# In[64]:


final_assignments=pd.DataFrame(final_assignments)
final_assignments


# In[67]:


final_assignments.to_excel('results.xls')
results=pd.read_excel('results.xls')
results


# In[ ]:




