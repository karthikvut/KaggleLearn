import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

games_data = pd.read_csv("C:\\Users\\M82828\\IdeaProjects\\KaggleLearn\\videogames\\vgsales.csv")
print(games_data.head())

#Relationship between platform and genre
platGenre = pd.crosstab(games_data.Platform, games_data.Genre)
platGenreTotal = platGenre.sum(axis=1).sort_values(ascending=False)
plt.figure(figsize=(10,10))
sns.barplot(y=platGenreTotal.index, x=platGenreTotal.values, orient='h')
plt.ylabel("Platform Genre")
plt.xlabel("Number of Games")
plt.show()

platPublisher = pd.crosstab(games_data.Platform, games_data.Publisher)
platPublisherTotal = platPublisher.sum(axis=1).sort_values(ascending=False)
plt.figure(figsize=(15,10))
plt.xlabel("No. of Games")
plt.ylabel("Platform Publisher")
sns.barplot(y=platPublisherTotal.index,x=platPublisherTotal.values,orient='h')
plt.show()

platGenre['Total'] = platGenre.sum(axis=1)
popPlatform = platGenre[platGenre['Total']>1000].sort_values(by='Total',ascending=False)
neededdata = popPlatform.loc[:,:'Strategy']
maxi = neededdata.values.max()
mini = neededdata.values.min()
popPlatformfinal = popPlatform.append(pd.DataFrame(popPlatform.sum(),columns=['total']).T,ignore_index=False)
sns.set(font_scale=1.0)
plt.figure(figsize=(10,10))
sns.heatmap(popPlatformfinal,vmin=mini,vmax=maxi,annot=True,fmt="d")
plt.xticks(rotation=90)
plt.show()

#GenreGroup Heatmap
GenreGroup = games_data.groupby(['Genre']).sum().loc[:,"NA_Sales":"Global_Sales"]
GenreGroup['NA_Sales%'] = GenreGroup['NA_Sales']/GenreGroup['Global_Sales']
GenreGroup['EU_Sales%'] = GenreGroup['EU_Sales']/GenreGroup['Global_Sales']
GenreGroup['JP_Sales%'] = GenreGroup['JP_Sales']/GenreGroup['Global_Sales']
GenreGroup['Other_Sales%'] = GenreGroup['Other_Sales']/GenreGroup['Global_Sales']

plt.figure(figsize=(10,10))
sns.set(font_scale=0.7)
plt.subplot(211)
sns.heatmap(GenreGroup.loc[:,'NA_Sales':'Other_Sales'],annot=True,fmt='.1f')
plt.title("Total sales by Genre")
plt.subplot(212)
sns.heatmap(GenreGroup.loc[:,'NA_Sales%':'Other_Sales%'], vmax=1, vmin=0, annot=True, fmt='.2%')
plt.title("Total percentage sales by Genre")
plt.show()


