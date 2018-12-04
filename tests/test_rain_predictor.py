import RainPredictor
import seaborn as sns
import matplotlib.pyplot as plt
import re


def test_rain_predictor_file_read():
    rain_predictor = RainPredictor.RainPredictor('../data/Weather Dataset_Filtered.csv')
    dataframe = rain_predictor.read_file(column_indices=[8, 10, 15, 14])
    dataframe = rain_predictor.rename_columns(dataframe, ['isRain', 'temperature', 'humidity', 'DewPointCelsius'])
    dataframe = rain_predictor.replace_data(dataframe, 'isRain', re.compile('^.*(RA|SN|DN|PL).*$'), 'Yes')
    dataframe = rain_predictor.replace_data(dataframe, 'isRain', re.compile('^(?!.*Yes).*$'), 'No')
    # Display the unique list of test data of probable predicted values
    # sns.countplot(y=dataframe['windspeed'], data=dataframe)
    # plt.show()
    (logistic, x_test, y_test) = rain_predictor.prediction_logic(dataframe[['humidity']+['temperature']+['DewPointCelsius']], dataframe[['isRain']])
    # Display the confusion matrix
    # c_matrix = rain_predictor.get_confusion_matrix(logistic, x_test, y_test)
    # rain_predictor.display_confusion_matrix(c_matrix)
    print(rain_predictor.get_rain_prediction(logistic, [[24.4, 90, 25]]))
    

test_rain_predictor_file_read()
