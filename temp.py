import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


data = pd.read_csv('healthcare-dataset-stroke-data.csv')

#Eξετάζουμε τους τύπους των μεταβλητών που έχει κάθε στήλη των δεδομένων μας.
print(data.info())
print('\n')

#Παρατηρούμε ότι υπάρχουν τιμές στην ηλικία που φέρουν δεκαδικό ψηφίο.
print('Minimum age: ' + str(data['age'].min()) + '\nMaximum age: ' + str(data['age'].max()) + '\n\n')

data['age'].hist(bins=82)

#Προχωρήσαμε σε στρογγυλοποίηση της ηλικίας, έχοντας κανένα δεκαδικό ψηφίο.
data['age'] = data['age'].round(0)
print('New minimum age: ' + str(data['age'].min()) + '\nNew maximum age: ' + str(data['age'].max()) + '\n\n')

data['age'].hist(bins=82)

#Για την πιό όμορφη παρουσίαση των δεδομένων, δημιουργήσαμε buckets μεγέθους 5 για την ηλικία.
data['shrink_ages'] = pd.cut(data['age'], np.arange(0, 86, 5))

#Έπειτα αναπαριστούμε την σχέση της ηλικίας με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plotage = data.groupby(['shrink_ages', 'stroke'])['shrink_ages'].count().unstack()
plotage.plot.bar()

#Εξετάζουμε την καπνιστική συνήθεια των ανθρώπων στα δεδομένα μας.
print('Smoking status\n' + str(data['smoking_status'].value_counts()) + '\n\n')

#Έπειτα αναπαριστούμε την σχέση της καπνιστικής συνήθειας με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plotsmoke = data.groupby(['smoking_status', 'stroke'])['smoking_status'].count().unstack()
plotsmoke.plot.bar()

#Εξετάζουμε το φύλο των ανθρώπων υπάρχουν στα δεδομένα μας.
print('Genders\n' + str(data['gender'].value_counts()) + '\n\n')

#Έπειτα αναπαριστούμε την σχέση του φύλου με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plotgender = data.groupby(['gender', 'stroke'])['gender'].count().unstack()
plotgender.plot.bar()

#Εξετάζουμε τον δείκτη μάζας των ανθρώπων στα δεδομένα μας.
print('Minimum BMI: ' + str(data['bmi'].min()) + '\nMaximum BMI: ' + str(data['bmi'].max()) + '\n\n')

#Για την πιό όμορφη παρουσίαση των δεδομένων, δημιουργήσαμε buckets μεγέθους 5 για τον δείκτη μάζας.
data['shrink_bmi'] = pd.cut(data['bmi'], np.arange(0, 101, 5))

#Έπειτα αναπαριστούμε την σχέση του δείκτη μάζας με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plotbmi = data.groupby(['shrink_bmi', 'stroke'])['shrink_bmi'].count().unstack()
plotbmi.plot.bar()

#Εξετάζουμε τα μέσα επίπεδα γλυκόζης των ανθρώπων στα δεδομένα μας.
print('Minimum average glucose level: ' + str(data['avg_glucose_level'].min()) + '\nMaximum average glucose level: ' + str(data['avg_glucose_level'].max()) + '\n\n')

#Για την πιό όμορφη παρουσίαση των δεδομένων, δημιουργήσαμε buckets μεγέθους 5 για τα μέσα επίπεδα γλυκόζης.
data['shrink_glucose'] = pd.cut(data['avg_glucose_level'], np.arange(0, 281, 10))

#Έπειτα αναπαριστούμε την σχέση των μέσων επιπέδων γλυκόζης με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plotglucose = data.groupby(['shrink_glucose', 'stroke'])['shrink_glucose'].count().unstack()
plotglucose.plot.bar()

#Εξετάζουμε την ύπαρξη υπέρτασης στους ανθρώπους που υπάρχουν στα δεδομένα μας.
print('Hypertension\n' + str(data['hypertension'].value_counts()) + '\n\n')

#Έπειτα αναπαριστούμε την σχέση της υπέρτασης με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plothypertension = data.groupby(['hypertension', 'stroke'])['hypertension'].count().unstack()
plothypertension.plot.bar()

#Εξετάζουμε την ύπαρξη καρδιακής πάθησης στους ανθρώπους που υπάρχουν στα δεδομένα μας.
print('Heart Disease\n' + str(data['heart_disease'].value_counts()) + '\n\n')

#Έπειτα αναπαριστούμε την σχέση της καρδιακής πάθησης με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plotheartdisease = data.groupby(['heart_disease', 'stroke'])['heart_disease'].count().unstack()
plotheartdisease.plot.bar()

#Εξετάζουμε το μέρος διαμονής των ανθρώπων που υπάρχουν στα δεδομένα μας.
print('Residence type\n' + str(data['Residence_type'].value_counts()) + '\n\n')

#Έπειτα αναπαριστούμε την σχέση του μέρους διαμονής με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plotresidence = data.groupby(['Residence_type', 'stroke'])['Residence_type'].count().unstack()
plotresidence.plot.bar()

#Εξετάζουμε την οικογενειακή κατάσταση των ανθρώπων που υπάρχουν στα δεδομένα μας.
print('Ever married\n' + str(data['ever_married'].value_counts()) + '\n\n')

#Έπειτα αναπαριστούμε της οικογενειακής κατάστασης με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plotmarried = data.groupby(['ever_married', 'stroke'])['ever_married'].count().unstack()
plotmarried.plot.bar()

#Εξετάζουμε την επαγγελματική κατάσταση των ανθρώπων που υπάρχουν στα δεδομένα μας.
print('Work type\n' + str(data['work_type'].value_counts()) + '\n\n')

#Έπειτα αναπαριστούμε την σχέση της επαγγελματικής κατάστασης με το εάν έχει υποστεί εγκεφαλικό επεισόδιο ή όχι.
plotwork = data.groupby(['work_type', 'stroke'])['work_type'].count().unstack()
plotwork.plot.bar()


#Αφαιρούμε τις περιττές για την συμπλήρωση τιμών στήλες των δεδομένων μας
not_used_cols = ['id', 'shrink_ages', 'shrink_bmi', 'shrink_glucose']
for i in not_used_cols:
    data = data.drop(i, axis=1)
    
#Φτιάχνουμε ένα αντίγραφο του μοντέλου που θα εφαρμόσουμε τις μεθόδους που μας ζητούνται.
data_new = data.copy()

#Τροποποιούμε τις τιμές των κατηγορικών δεδομένων για την καλύτερη διαχείρισή τους.
data_new['smoking_status'] = data_new['smoking_status'].map({'Unknown' : np.nan, 'never smoked' : 0,  
                                                         'formerly smoked' : 1, 'smokes' : 2})
data_new['gender'] = data_new['gender'].map({'Male' : 0, 'Female' : 1})
data_new['ever_married'] = data_new['ever_married'].map({'No' : 0, 'Yes' : 1})
data_new['work_type'] = data_new['work_type'].map({'Private' : 0, 'Self-employed' : 1, 'children' : 2,
                                                   'Govt_job' : 3, 'Never_worked' : 4})
data_new['Residence_type'] = data_new['Residence_type'].map({'Urban' : 0, 'Rural' : 1})

#Μελετάμε τις ελλειπείς τιμές.
print('Missing values from dataset\n' + str(data_new.isna().sum()))

#Δεδομένα έπειτα από αφαίρεση της στήλης που περιέχει μια τιμή NaN.
data_drop = data_new.dropna()

#Δεδομένα έπειτα από την συμπλήρωση των τιμών NaN με το μέσο όρο των υπόλοιπων τιμών της στήλης.
data_replace_mean = data_new.copy()

#Aντικαθηστούμε την κάθε NaN τιμή με τον μέσο όρο της αντίστοιχης στήλης.
data_replace_mean['smoking_status'] = data_replace_mean['smoking_status'].replace(np.nan, data_replace_mean['smoking_status'].mean())
data_replace_mean['bmi'] = data_replace_mean['bmi'].replace(np.nan, data_replace_mean['bmi'].mean())

#Δεδομένα έπειτα από συμπλήρωση τιμών NaN με kNN μέθοδο
data_knn = data_new.copy()

#Scaling των δεδομένων μας.
scaler = MinMaxScaler()
data_knn = pd.DataFrame(scaler.fit_transform(data_knn), columns = data_knn.columns)

#Χρήση του kNN imputer για τον υπολογισμό των άγνωστων τιμών μας.
imputer = KNNImputer(n_neighbors=5)
data_knn = pd.DataFrame(imputer.fit_transform(data_knn),columns = data_knn.columns)

cat_values = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']