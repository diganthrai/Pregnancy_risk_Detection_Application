
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('Backend/ML/Random_Forest/Maternal Health Risk Data Set.csv')

#Exploratory Data Analysis
upperlimit=df["Age"].mean()+1.5*df["Age"].std()
lowerlimit=18
print("upper limit",upperlimit)
print("lower limit", lowerlimit)

# remove outliers
new_df=df.loc[(df['Age']<upperlimit)&(df['Age']>=lowerlimit)]
print("Before",len(df))
print("After",len(new_df))
print("Outliers",len(df)-len(new_df))


new_df['RiskLevel'].replace({"high risk":"1","mid risk":"1","low risk":"0"}, inplace=True)
new_df['RiskLevel'] = new_df['RiskLevel'].astype(float)
new_df.BS=(new_df.BS*18)
print(new_df)
new_df.to_csv('Backend/ML/LSTM/New_Maternal.csv', index=False)
new_df.to_csv('Backend/ML/Random_Forest/New_Maternal.csv', index=False)

