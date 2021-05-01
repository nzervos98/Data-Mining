import numpy as np
import matplotlib as plt
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
# Στρογγυλοποίηση  των ηλικιών. 
data['age'] = data['age'].round(0)


# Δημιουργία νεου atribute για εξάληψη unknown 
data['smoking_status_new'] = data['smoking_status'].map({'Unknown' : np.nan, 'never smoked' : 0,  
                                                         'formerly smoked' : 1, 'smokes' : 2})


# Αφαίρεση σειράς. 
droped_data = data.dropna()


# Συμπλήρωση με τον μέσο όρο.  
m = np.mean(data['bmi'])
m = round(m,1)
replaced_data = data.copy()
replaced_data['bmi'] = replaced_data['bmi'].replace(np.nan,m)
k  = np.mean(replaced_data['smoking_status_new'])
replaced_data['smoking_status_new'] = replaced_data['smoking_status_new'].replace(np.nan,k)


# Συμπλήρωση με linear regression για των υπολογισμό των missing values του bmi
lr_data = data.copy()

cols = ["id", "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"] 
df = lr_data[cols]
test_df = df[df["bmi"].isnull()] # Επέστρεψε ενα df με τις γραμμές στις οποίες το bmi = nul.
df = df.dropna() # Αφαίρεση ολων των γραμμών που έχουν nul

Y_train = df["bmi"] 
X_train = df.drop("bmi", axis=1)
X_test = test_df.drop("bmi", axis=1)

# Train και Prediction για τις missing values του bmi
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

# Αντικατάσταση των missing values με τις predicted για το bmi
lr_data.loc[data.bmi.isnull(), 'bmi'] = Y_pred

# Συμπλήρωση με linear regression για των υπολογισμό των missing values του smoking_status_new
cols = ["id", "age", "hypertension", "heart_disease", "avg_glucose_level", "stroke", "smoking_status_new"] 
df = lr_data[cols]
test_df = df[df["smoking_status_new"].isnull()] # Επέστρεψε ενα df με τις γραμμές στις οποίες το smoking_status_new = nul.
df = df.dropna() # Αφαίρεση ολων των γραμμών που έχουν nul

Y_train = df["smoking_status_new"]
X_train = df.drop("smoking_status_new", axis=1)
X_test = test_df.drop("smoking_status_new", axis=1)

# Train και Prediction για τις missing values του smoking_status_new
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

# Αντικατάσταση των missing values με τις predicted για το smoking_status_new
lr_data.loc[data.smoking_status_new.isnull(), 'smoking_status_new'] = Y_pred