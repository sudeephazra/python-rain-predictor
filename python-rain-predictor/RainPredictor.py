import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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

    _model = 'RainPredictor.mdl'
    _training_data_set = '../data/Weather Dataset_Filtered.csv'
    # _training_data_set = '../data/sample.csv'

    # Initialize the class with a optional model file
    # If the model file is provided then this is the model file to be used, else use a default model file name
    def __init__(self, model_file=None, training_data_set=None):
        if model_file is None:
            self.model_file = self._model
        else:
            self.model_file = model_file
        if training_data_set is None:
            self.training_data_set = self._training_data_set
        else:
            self.training_data_set = training_data_set
        pandas.options.mode.chained_assignment = None

    # Get the HOME folder of the current logged in user.
    # This returns a platform independent location of the HOME folder
    @staticmethod
    def get_home_folder():
        home_folder = os.path.expanduser('~')
        return home_folder

    # Get the absolute PATH of the model file
    def get_trained_model_file(self):
        trained_file = os.path.join(self.get_home_folder(), self.model_file)
        return trained_file

    # Check if the model is already trained
    # This is done by checking the existence of the saved model file
    def get_model_trained_status(self):
        # if trained file exists that means model is trained
        if os.path.isfile(self.get_trained_model_file()) is True:
            return True
        else:
            return False

    def save_trained_model(self, logistic):
        dump(logistic, self.get_trained_model_file())

    def load_trained_model(self):
        data_frame = load(self.get_trained_model_file())
        return data_frame

    def forget_training(self):
        trained_model = self.get_trained_model_file()
        if os.path.exists(trained_model):
            try:
                os.remove(trained_model)
            except OSError as e:
                print("Error: %s - %s." % (e.trained_model, e.strerror))

    # TODO: column_indices should default to all columns
    def read_training_data_set(self, column_indices, headers=0):
        data_file = self.training_data_set
        data_frame = pandas.read_csv(data_file, header=headers, usecols=column_indices)
        return data_frame

    @staticmethod
    def rename_columns(data_frame, column_names):
        data_frame.columns = column_names
        return data_frame

    @staticmethod
    def replace_data(data_frame, column_name, old_string, new_string):
        data_frame[column_name].replace(to_replace=old_string, value=new_string, regex=True, inplace=True)
        return data_frame

    # TODO: This method should encapsulate the entire prediction flow
    def prediction_logic(self, x_dataframe, y_dataframe):
        if self.is_model_trained() is True:
            dataframe = self.load_trained_model()
        else:
            self.train_model_logistic_regression(x_dataframe, y_dataframe)

    def train_model_logistic_regression(self, x_dataframe, y_dataframe):
        x_train, x_test, y_train, y_test = train_test_split(x_dataframe,
                                                            y_dataframe, random_state=0, test_size=0.25)
        logistic = LogisticRegression(solver='lbfgs')
        logistic.fit(x_train, y_train.values.ravel())
        self.save_trained_model(logistic)
        return logistic, x_test, y_test

    def train_model_naive_bayes(self, x_dataframe, y_dataframe):
        x_train, x_test, y_train, y_test = train_test_split(x_dataframe,
                                                            y_dataframe, random_state=0, test_size=0.25)
        gaussian = GaussianNB()
        gaussian.fit(x_train, y_train.values.ravel())
        self.save_trained_model(gaussian)
        return gaussian, x_test, y_test

    @staticmethod
    def get_confusion_matrix(logistic, x_test, y_test):
        predict = logistic.predict(x_test)
        c_matrix = confusion_matrix(y_test, predict)
        return c_matrix

    @staticmethod
    def display_confusion_matrix(confusion_matrix):
        plt.rc("font", size=14)
        sns.set(style="white")
        sns.set(style="whitegrid", color_codes=True)
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @staticmethod
    def get_rain_prediction(logistic, x_test):
        predict = logistic.predict(x_test)
        return predict.item(0)

    @staticmethod
    def get_rain_prediction_accuracy(logistic, x_test):
        predict = logistic.predict_proba(x_test)
        if predict.item(0) > predict.item(1):
            return predict.item(0)
        else:
            return predict.item(1)
