import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Schematics
# 1. Read the data file
# 2. Select specific columns only
# 3. Rename column headers as required
# 4. Replace values if required
# 5. Apply algorithm and predict

# 1. Read data file
data = pandas.read_csv('../data/Weather Dataset_Filtered.csv', header=0, usecols=[8, 10, 15])
data = data.dropna()
# print(data.shape)
# print(list(data.columns))
# 2. Select specific columns only
# data_cleansed = data[['WeatherType', 'DryBulbCelsius', 'RelativeHumidity']]
data_cleansed = data
# 3. Rename column headers as required
data_cleansed.columns = ['isRain', 'temperature', 'humidity']
# 4. Replace values if required
pandas.options.mode.chained_assignment = None  # default='warn'
data_cleansed['isRain'].replace(to_replace=r'^.*(RA|SN|DN|PL).*$', value=r'Yes', regex=True, inplace=True)
data_cleansed['isRain'].replace(to_replace=r'^(?!.*Yes).*$', value=r'No', regex=True, inplace=True)
# print(data_cleansed.shape)
# print(data_cleansed)

#
X_train, X_test, y_train, y_test = train_test_split(data_cleansed[['humidity']+['temperature']], data_cleansed[['isRain']], random_state=0, test_size=0.25)

# sns.countplot(y=data_cleansed['isRain'], data=data_cleansed)
# plt.show()


logistic = LogisticRegression(solver='lbfgs')
logistic.fit(X_train, y_train.values.ravel())
predict = logistic.predict(X_test)
# print(X_test)
cm1 = confusion_matrix(y_test, predict)
# print(cm1)
print(logistic.predict([[24.4, 90]]))
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logistic.score(X_test, y_test)))
#
# # Display the plot
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(6, 6))
sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.show()



