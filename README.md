# -Learn-how-to-clean-and-prepare-raw-data-for-ML.
# Titanic Survival Prediction 🚢

A machine learning project using the Titanic dataset to predict passenger survival. This project demonstrates data preprocessing, feature engineering, and predictive modeling using Python and popular data science libraries.

## Dataset

The dataset used is a modified version of the Titanic dataset containing features such as:

- `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex`: Gender
- `Age`: Age of passenger
- `SibSp`: # of siblings/spouses aboard
- `Parch`: # of parents/children aboard
- `Fare`: Ticket fare
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- 
##  Preprocessing Steps

1. **Handle Missing Values**
   - `Age`: Filled with median
   - `Fare`: Filled with mean
   - Dropped columns with excessive missing data (`Cabin`, `Ticket`) or irrelevant (`PassengerId`, `Name`)

2. **Encoding**
   - `Sex`: Converted to numeric (`male` → 0, `female` → 1)
   - `Embarked`: One-hot encoded (using `pd.get_dummies`)

3. **Feature Selection**
   - Selected only meaningful features for the model

**Model Building (Optional)**

You can use models like:
- Logistic Regression
- Random Forest
- XGBoost

Example:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
