from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

df = pd.read_csv('Irisflower.csv')

x = df.iloc[:,1:-1]

LE = LabelEncoder()

y = LE.fit_transform(df.iloc[:,-1])

LR = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=10)

LR.fit(x_train,y_train)
print("Model Trained successfully...\n")
y_pred = LR.predict(x_test)

print("RMSE: ",root_mean_squared_error(y_test,y_pred))
print("MSE: ", mean_squared_error(y_test,y_pred))
print("r2_score: ", r2_score(y_test,y_pred))

with open("LR.pkl",'wb') as f:
    pickle.dump(LR,f)

with open("LE.pkl","wb") as f:
    pickle.dump(LE,f)