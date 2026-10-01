import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Sample data for Smartphone Addiction Prediction
data = {
 'ScreenTime': [7,2,5,8,3,6,4,2.5,9,3.5],
 'SleepHours': [5,8,6,4,7,5,7,8,3,7],
 'PhoneChecks': [90,20,60,110,30,80,45,25,130,35],
 'Addiction': ['High','Low','Medium','High','Low','High','Medium','Low','High','Low']
}
df = pd.DataFrame(data)
X = df[['ScreenTime','SleepHours','PhoneChecks']]
y = df['Addiction']

model = RandomForestClassifier()
model.fit(X,y)
print(model.predict([[7,5,90]]))
