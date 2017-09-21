import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

moviedata = pd.read_csv("C:\\Users\\M82828\\Desktop\\Python HW\\movie_metadata.csv")
#print(moviedata.head())

str_col_list = []
for colName,colVal in moviedata.iteritems():
    if type(colVal[1]) == str:
        str_col_list.append(colName)
num_col_list = moviedata.columns.difference(str_col_list)

moviedata_num = moviedata[num_col_list]
#print(moviedata_num.head())

moviedata_num = moviedata_num.fillna(value=0,axis=1)

from sklearn.preprocessing import StandardScaler

moviedata_num_values=moviedata_num.values
std_moviedata_num = StandardScaler().fit_transform(moviedata_num_values)

#Matplotlib figure
f, ax = plt.subplots(figsize=(12,10))
plt.title('Correlation of Movie Features')
sns.heatmap(moviedata_num.astype(float).corr(),linewidths=0.25,vmax=1.0,square=True,cmap="YlGnBu",linecolor='black',annot=True)
#plt.show()

mean_vector = np.mean(std_moviedata_num,axis=0)
covariance_matrix = np.cov(std_moviedata_num.T)
eigenVals, eigenVecs = np.linalg.eig(covariance_matrix)

eigenPairs = [(np.abs(eigenVals),eigenVecs[:,i]) for i in range(len(eigenVals))]
#print("Eigen Pairs:",eigenPairs[0])
eigenPairs.sort(key=lambda x:x[0].all(), reverse=True)

total = sum(eigenVals)
var_explained = [(i/total) * 100 for i in sorted(eigenVals,reverse=True)]
cum_var_explained = np.cumsum(var_explained)

plt.figure(figsize=(10,5))
plt.bar(range(len(eigenVals)), var_explained, alpha=0.333, align='center', label='Individual explained variance', color='g')
plt.step(range(len(eigenVals)), cum_var_explained, where='mid', label='Cumulative explained variance', color='r')
plt.ylabel('Explained Variance')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=9)
pca_comp = pca.fit_transform(std_moviedata_num)
print("PCA COMPONENTS:",pca_comp, pca_comp.shape)
plt.figure(figsize=(9,7))
plt.scatter(pca_comp[:,0],pca_comp[:,1],c='goldenrod',alpha=0.5)
plt.ylim(-10,30)
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
X_clustered = kmeans.fit_predict(pca_comp)

LABEL_COLOR_MAP = {0:'r',1:'g',2:'b',3:'c', 4:'m', 5:'k', 6:'y', 7:'0.75', 8:'0.63'}
label_color = [LABEL_COLOR_MAP[i] for i in X_clustered]

plt.figure(figsize=(10,10))
plt.scatter(pca_comp[:,0], pca_comp[:,2], c=label_color, alpha=0.5)
plt.show()
