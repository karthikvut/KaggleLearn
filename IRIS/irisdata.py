import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect("C:\\Users\\M82828\\IdeaProjects\\KaggleLearn\\IRIS\\database.sqlite")
irisdata = pd.read_sql("SELECT * FROM IRIS",conn)
print(irisdata.head())

print(irisdata.Species.value_counts())

irisdata.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm")
plt.show()

sns.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=irisdata,size=5)
plt.show()

sns.FacetGrid(irisdata,hue='Species',size=5).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()
plt.show()

sns.boxplot(x="Species",y="PetalLengthCm",data=irisdata)
plt.show()

ax=sns.boxplot(x="Species",y="PetalWidthCm",data=irisdata)
ax=sns.stripplot(x="Species",y="PetalWidthCm",data=irisdata,jitter=True,edgecolor="gray")
plt.show()

sns.pairplot(data=irisdata.drop("Id",axis=1),hue="Species",size=3)
plt.show()