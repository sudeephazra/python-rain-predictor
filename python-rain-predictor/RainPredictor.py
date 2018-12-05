import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import os
from joblib import dump, load

#
# Initialize the model
# check if the model is already trained and saved
# if yes, load the trained model, else train th model from the data file provided
# display the predicted value accordingly
#
class RainPredictor:

    def __init__(self, file):
        self.data_file = file
        pandas.options.mode.chained_assignment = None

    def get_home_folder(self):
        home_folder = os.path.expanduser('~')
        return home_folder

    def get_trained_file(self):
        trained_file = os.path.join(self.get_home_folder(), "LogisticRegression.mdl")
        return trained_file

    # Check if the model is already trained
    # This is done by checking the existence of the saved model file
    def is_model_trained(self):
        # if trained file exists that means model is trained
        if os.path.isfile(self.get_trained_file()) is True:
            return True
        else:
            return False

    def read_file(self, column_indices, headers=0):
        data_frame = pandas.read_csv(self.data_file, header=headers, usecols=column_indices) # column_indices should default to all columns
        return data_frame

    def rename_columns(self, data_frame, column_names):
        data_frame.columns = column_names
        return data_frame

    def replace_data(self, data_frame, column_name, old_string, new_string):
        data_frame[column_name].replace(to_replace=old_string, value=new_string, regex=True, inplace=True)
        return data_frame

    def train_model(self, x_dataframe, y_dataframe):
        x_train, x_test, y_train, y_test = train_test_split(x_dataframe,
                                                            y_dataframe, random_state=0, test_size=0.25)
        logistic = LogisticRegression(solver='lbfgs')
        logistic.fit(x_train, y_train.values.ravel())
        self.save_trained_model(logistic)
        return logistic, x_test, y_test

    def prediction_logic(self, x_dataframe, y_dataframe):
        if self.is_model_trained() is True:
            dataframe = load("LogisticRegression.mdl")
        else:
            self.train_model(x_dataframe, y_dataframe)




    def save_trained_model(self, logistic):
        dump(logistic, self.get_trained_file())


    def get_confusion_matrix(self, logistic, x_test, y_test):
        predict = logistic.predict(x_test)
        c_matrix = confusion_matrix(y_test, predict)
        return c_matrix

    def display_confusion_matrix(self, confusion_matrix):
        plt.rc("font", size=14)
        sns.set(style="white")
        sns.set(style="whitegrid", color_codes=True)
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def get_rain_prediction(self, logistic, x_test):
        predict = logistic.predict(x_test)
        return predict
