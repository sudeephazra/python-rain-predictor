import RainPredictor
import seaborn as sns
import matplotlib.pyplot as plt
import re


# def test_rain_predictor_file_read():
#     rain_predictor = RainPredictor.RainPredictor('../data/Weather Dataset_Filtered.csv')
#     dataframe = rain_predictor.read_file(column_indices=[8, 10, 14, 15])
#     dataframe = rain_predictor.rename_columns(dataframe, ['isRain', 'temperature', 'DewPointCelsius', 'humidity'])
#     dataframe = rain_predictor.replace_data(dataframe, 'isRain', re.compile('^.*(RA|SN|DN|PL).*$'), 'Yes')
#     dataframe = rain_predictor.replace_data(dataframe, 'isRain', re.compile('^(?!.*Yes).*$'), 'No')
#     # # Display the unique list of test data of probable predicted values
#     # # sns.countplot(y=dataframe['windspeed'], data=dataframe)
#     # # plt.show()
#     (logistic, x_test, y_test) = rain_predictor.prediction_logic(dataframe[['humidity']+['temperature']+['DewPointCelsius']], dataframe[['isRain']])
#     # # Display the confusion matrix
#     # # c_matrix = rain_predictor.get_confusion_matrix(logistic, x_test, y_test)
#     # # rain_predictor.display_confusion_matrix(c_matrix)
#     print(rain_predictor.get_rain_prediction(logistic, [[25, 10, 80]]))


def test_code_modularization():
    rain_predictor = RainPredictor.RainPredictor()
    rain_predictor.forget_training()
    if rain_predictor.get_model_trained_status() is False:
        print("Training...")
        data_frame = rain_predictor.read_training_data_set(column_indices=[8, 10, 15])
        data_frame = rain_predictor.rename_columns(data_frame, ['isRain', 'temperature', 'humidity'])
        data_frame = rain_predictor.replace_data(data_frame, 'isRain', re.compile('^.*(RA|SN|DN|PL).*$'), 'Yes')
        data_frame = rain_predictor.replace_data(data_frame, 'isRain', re.compile('^(?!.*Yes).*$'), 'No')
        sns.countplot(y=data_frame['isRain'], data=data_frame)
        plt.show()
        (logistic, x_test, y_test) = rain_predictor.train_model(
            data_frame[['temperature'] + ['humidity']], data_frame[['isRain']])
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logistic.score(x_test, y_test)))
        # Display the confusion matrix
        c_matrix = rain_predictor.get_confusion_matrix(logistic, x_test, y_test)
        rain_predictor.display_confusion_matrix(c_matrix)
    else:
        print("Reusing trained data....")
        logistic = rain_predictor.load_trained_model()

    print(rain_predictor.get_rain_prediction(logistic, [[32, 98]]))
