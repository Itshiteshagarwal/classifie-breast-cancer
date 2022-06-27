import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
df = pd.read_csv('Copy of breast-cancer-wisconsin.csv')
array=df.values
X = df.iloc[:, :3]
y = df.iloc[:, 3] 
d=X.head()
l=y.head()
d_train,d_validation,l_train,l_validation=train_test_split(d,l,test_size=0.2,random_state=1)
model=SVC(gamma='auto')
model.fit(d_train,l_train)
prediction=model.predict(d_validation)
print(accuracy_score(l_validation,prediction))
print(confusion_matrix(l_validation,prediction))
print(classification_report(l_validation,prediction))