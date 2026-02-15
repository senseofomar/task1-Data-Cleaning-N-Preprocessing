import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_csv('Titanic-Dataset.csv')

print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

#removing outliers first quartile 25th and third quartile 75th percentiles
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

#Rule
#Q1 - 1.5*IQR   lesser
#Q3 + 1.5*IQR   greater

df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

#df = df[((df['Fare'] > (Q1 - 1.5 * IQR)) & (df['Fare'] < (Q3 + 1.5 * IQR)))]

print("Preprocessing Complete. Cleaned data shape:", df.shape)
df.to_csv('cleaned_titanic.csv', index=False)