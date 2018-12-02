import pandas
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy
# from sklearn import preprocessing
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Read data file
# data = pandas.read_csv('../data/sample.csv', header=0)
# data = data.dropna()
# print(data.shape)
# print(list(data.columns))
# sns.countplot(x='DryBulbCelsius',data=data, palette='hls')
# plt.show()

# Schemantics
# 1. Read the data file
# 2. Select specific columns only
# 3. Rename column headers as required
# 4. Replace values if required
# 5. Apply algorithm and predict

# 1. Read data file
data = pandas.read_csv('../data/Weather Dataset_Filtered.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))
# 2. Select specific columns only
data_cleansed = data[['WeatherType', 'DryBulbCelsius', 'RelativeHumidity']]
# 3. Rename column headers as required
data_cleansed.columns = ['isRain', 'temperature', 'humidity']
# 4. Replace values if required
pandas.options.mode.chained_assignment = None  # default='warn'
data_cleansed['isRain'].replace(to_replace=r'-(RA|SN|DN|PL).*$', value=r'Yes', regex=True, inplace=True)
data_cleansed['isRain'].replace(to_replace=r'^(?!.*Yes).*$', value=r'No', regex=True, inplace=True)
# print(data_cleansed.shape)
# print(data_cleansed)


# x = data_cleansed.iloc[:, 2:]
# y = data_cleansed.iloc[:, 1:]
# X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)
# print(X_train.shape)
# print(y)
# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)
# from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

# from sklearn import svm, datasets
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix #for using confusion matrix###
#
logistic = LogisticRegression()
logistic.fit(data_cleansed[['temperature']+['humidity']], data_cleansed[['isRain']])
predict = logistic.predict(data_cleansed[['temperature']+['humidity']])
# Use score method to get accuracy of model
cm1 = confusion_matrix(data_cleansed[['isRain']], predict)
print(cm1)
total1 = sum(sum(cm1))
# print(total1)
accuracy1 = (cm1[0, 0] + cm1[1, 1] + cm1[2, 2] + cm1[3, 3] + cm1[4, 4] + cm1[5, 5])/total1
print(accuracy1)

# plt.figure(figsize=(9,9))
# sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
# plt.ylabel('Actual label');
# plt.xlabel('Predicted label');
# all_sample_title = 'Accuracy Score: {0}'.format(score)
# plt.title(all_sample_title, size = 15);


