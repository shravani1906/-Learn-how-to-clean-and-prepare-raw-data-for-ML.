import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode categorical
    df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Normalize numeric
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    # Remove outliers
    z_scores = np.abs(stats.zscore(df[['Age', 'Fare']]))
    df = df[(z_scores < 3).all(axis=1)]

    return df

if __name__ == '__main__':
    clean_df = load_and_preprocess('dataset.csv')
    print(clean_df)
