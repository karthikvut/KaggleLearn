import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\M82828\\Downloads\\creditcard.csv")
print(data.head())

counts = pd.value_counts(data.Class, sort=True).sort_index()
plt.title("Pie chart of Class Data")
plt.pie(counts,labels=data.Class.unique())
plt.show()

from sklearn.preprocessing import StandardScaler
data['normAmt'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data=data.drop(['Time','Amount'],axis = 1)
print(data.head())

X = data.ix[:,data.columns != 'Class']
Y = data.ix[:, data.columns == 'Class']

fraud_count = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class==1].index)

non_fraud_indices = data[data.Class==0].index

random_non_fraud_indices = np.random.choice(non_fraud_indices, fraud_count, replace=False)
random_non_fraud_indices = np.array(random_non_fraud_indices)

sample_indices = np.concatenate([fraud_indices,random_non_fraud_indices])
sample_data = data.iloc[sample_indices,:]

X_sample = sample_data.ix[:,sample_data.columns !='Class']
Y_sample = sample_data.ix[:,sample_data.columns == 'Class']

print("Percentage of non fraud transactions:", len(sample_data[sample_data.Class == 0])/len(sample_data))
print("Percentage of Fraud transactions:",len(sample_data[sample_data.Class==1])/len(sample_data))
print("Total no. of transactions:",len(sample_data))

from sklearn.cross_validation import train_test_split

#Whole dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
print("Number of transactions in train dataset:",len(X_train))
print("Number of transaction in test dataset:",len(X_test))
print("Total Number of transactions in train and test:",len(X_train)+len(X_test))


#Sample dataset
X_train_sample, X_test_sample, Y_train_sample, Y_test_sample = train_test_split(X_sample,Y_sample, test_size=0.3,random_state=0)
print("")
print("Number of transactions in sample train dataset:",len(X_train_sample))
print("Number of transactions in sample test dataset:",len(X_test_sample))
print("Total number of transactions in dataset:",len(X_train_sample) + len(X_test_sample))

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, recall_score, classification_report

def print_KFold_Scores(X_train_data,Y_train_data):
    fold = KFold(len(Y_train_data),5,shuffle=False)
    print(fold)
    c_param_range = [0.01,0.1,1,10,100]
    results_table = pd.DataFrame(index=range(len(c_param_range),2),columns=['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    j=0
    for c_param in c_param_range:
        print('-----------')
        print('C parameter', c_param)
        print('------------')
        print('')

        recall_accs= []
        for iteration, indices in enumerate(fold,start=1):
            lr=LogisticRegression(C=c_param, penalty='l1')
            lr.fit(X_train_data.iloc[indices[0],:],Y_train_data.iloc[indices[0],:].values.ravel())
            Y_pred_sample = lr.predict(X_train_sample.iloc[indices[1],:].values)
            recall_acc = recall_score(Y_train_data.iloc[indices[1],:].values, Y_pred_sample)
            recall_accs.append(recall_acc)
            print('Iteration:', iteration, ':Recall Score = ',recall_acc)

    results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
    j += 1
    print('')
    print('Mean Recall score',np.mean(recall_accs))
    print('')
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    print('*************************************************************')
    print('Best model to choose from CV is with C param =',best_c)
    print('**************************************************************')

    return best_c

best_c = print_KFold_Scores(X_train_sample,Y_train_sample)





