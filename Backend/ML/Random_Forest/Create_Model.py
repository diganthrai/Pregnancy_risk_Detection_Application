from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd 
import pickle


df = pd.read_csv('Backend/ML/Random_Forest/New_Maternal.csv')

train , test = train_test_split(df,test_size=0.15,random_state=35)
print(f"No. of training examples:{train.shape[0]}")
print(f"No. of testing examples:{test.shape[0]}")

y_test= test['RiskLevel']
x_test= test.drop('RiskLevel', axis=1)

x_train= train.drop('RiskLevel',axis=1)
y_train= train['RiskLevel']

random_model=RandomForestClassifier(n_estimators=50)
random_model.fit(x_train,y_train)
random_predict=random_model.predict(x_test)
print("Accuracy=",metrics.accuracy_score(y_test,random_predict))

with open('Backend/ML/Random_Forest/model.pkl', 'wb') as f:
    pickle.dump(random_model, f)

