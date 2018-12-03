import pandas
import seaborn as sns
import matplotlib.pyplot as plt


class RainPredictor:

    def __init__(self, file):
        self.data_file = file
        pandas.options.mode.chained_assignment = None

    def read_file(self, column_indices, headers=0):
        data_frame = pandas.read_csv(self.data_file, header=headers, usecols=column_indices)
        return data_frame

    def rename_columns(self, data_frame, column_names):
        data_frame.columns = column_names
        return data_frame

    def replace_data_regex(self, data_frame, column_name, old_string, new_string):
        data_frame[column_name].replace(to_replace=old_string, value=new_string, regex=True, inplace=True)
        return data_frame

    def display_confusion_matrix(self, confusion_matrix):
        plt.rc("font", size=14)
        sns.set(style="white")
        sns.set(style="whitegrid", color_codes=True)
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
