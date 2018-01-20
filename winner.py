import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display
#from numpy import array
#from keras.utils import to_categorical

data=pd.read_csv('E0-1617.csv')

display(data.head())

X=data.iloc[:,2:21].values
y=data.iloc[:,21:].values

team_namedict=pd.DataFrame({'Hteams':[x for x in X[:,0]],'Ateams':[x for x in X[:,1]]})

from sklearn.preprocessing import LabelEncoder #, OneHotEncoder
LabelEncoder_hteam=LabelEncoder()
LabelEncoder_ateam=LabelEncoder()
LabelEncoder_htr=LabelEncoder()
X[:,0]=LabelEncoder_hteam.fit_transform(X[:,0])
X[:,1]=LabelEncoder_ateam.fit_transform(X[:,1])
X[:,6]=LabelEncoder_htr.fit_transform(X[:,6])

LabelEncoder_res=LabelEncoder()
y=LabelEncoder_res.fit_transform(y)

team_numbdict=pd.DataFrame({'Hteams':[x for x in X[:,0]],'Ateams':[x for x in X[:,1]]})

Hteam_name=[i for i in team_namedict['Hteams']]
Ateam_name=[i for i in team_namedict['Ateams']]

Hteam_numb=[i for i in team_numbdict['Hteams']]
Ateam_numb=[i for i in team_numbdict['Ateams']]

H_nametonumb=[]
A_nametonumb=[]

for i in zip(Hteam_numb,Hteam_name):
    H_nametonumb.append(i)
    
for i in zip(Ateam_numb,Ateam_name):
    A_nametonumb.append(i)

#y=array(y)
#y=to_categorical(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

clf_A=LogisticRegression(random_state=42)
clf_B=SVC(random_state =9,kernel='linear')
clf_C=xgb.XGBClassifier(seed=82)

A=clf_A.fit(X_train,y_train)
B=clf_B.fit(X_train,y_train)
C=clf_C.fit(X_train,y_train)


print("LR")
print(LabelEncoder_res.inverse_transform(A.predict(X_test)))

print("SVM")
print(LabelEncoder_res.inverse_transform(B.predict(X_test)))

print("XGB")
print(LabelEncoder_res.inverse_transform(C.predict(X_test)))

print("Actual")
print(LabelEncoder_res.inverse_transform(y_test))
