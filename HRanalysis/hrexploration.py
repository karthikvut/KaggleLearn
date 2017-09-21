import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

hrdata = pd.read_csv("HR_comma_sep.csv")
# print(hrdata.head())
# print(hrdata.describe())
# print(hrdata.salary.unique())
# print(hrdata.sales.unique())

def replace_salary(x):
    if x == 'low':
        return 0
    elif x == 'medium':
        return 1
    else:
        return 2

def replace_sales(x):
    if x == 'sales':
        return 0
    elif x == 'accounting':
        return 1
    elif x == 'hr':
        return 2
    elif x == 'technical':
        return 3
    elif x == 'support':
        return 4
    elif x == 'management':
        return 5
    elif x == 'IT':
        return 6
    elif x == 'product_mng':
        return 7
    elif x == 'marketing':
        return 8
    else:
        return 9

hrdata['salary'] = hrdata['salary'].apply(lambda x:replace_salary(x))
hrdata['sales'] = hrdata['sales'].apply(lambda x:replace_sales(x))

#print(hrdata.head())

plt.figure(figsize=(10,10))
sns.heatmap(data=hrdata.corr(),linewidths=0.5,linecolor='black',cmap="BuGn",annot=True)
plt.show()

hrdata_left = hrdata[hrdata['left']==1]
print(hrdata_left.head())

def plot_hists():

    fig, axes = plt.subplots(3, 3)
    axes = axes.ravel()
    cols = hrdata_left.columns.tolist()
    del cols[-4]
    for i,ax in enumerate(axes):
        ax.hist(hrdata_left[cols[i]],color='red')
        ax.set_title(cols[i])
        # ax.set_xlabel(cols[i])
        # ax.set_ylabel('left')
    plt.tight_layout()
    plt.show()


plot_hists()

hrdata_left_good = hrdata_left[(hrdata_left['last_evaluation'] >= 0.70) & (hrdata_left['time_spend_company'] >=4) & (hrdata_left['number_project'] >5) ]
print("Number of people left:",len(hrdata_left))
plt.figure(figsize=(10,10))
sns.heatmap(data=hrdata_left_good.corr(),linewidths=0.5,linecolor='black',cmap="BuGn",annot=True)
plt.show()