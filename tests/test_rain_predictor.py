import RainPredictor
import seaborn as sns
import matplotlib.pyplot as plt
import re


def test_rain_logistic_regression():
    rain_predictor = RainPredictor.RainPredictor()
    # rain_predictor.forget_training()  # This will train the model on each execution
    if rain_predictor.get_model_trained_status() is False:
        print("\nTraining...")
        data_frame = rain_predictor.read_training_data_set(column_indices=[8, 10, 15])
        data_frame = rain_predictor.rename_columns(data_frame, ['isRain', 'temperature', 'humidity'])
        data_frame = rain_predictor.replace_data(data_frame, 'isRain', re.compile('^.*(RA|SN|DN|PL).*$'), 'Yes')
        data_frame = rain_predictor.replace_data(data_frame, 'isRain', re.compile('^(?!.*Yes).*$'), 'No')
        sns.countplot(y=data_frame['isRain'], data=data_frame)
        plt.show()
        (logistic, x_test, y_test) = rain_predictor.train_model_logistic_regression(
            data_frame[['temperature'] + ['humidity']], data_frame[['isRain']])
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logistic.score(x_test, y_test)))
        # Display the confusion matrix
        c_matrix = rain_predictor.get_confusion_matrix(logistic, x_test, y_test)
        rain_predictor.display_confusion_matrix(c_matrix)
    else:
        print("\nReusing trained data....")
        logistic = rain_predictor.load_trained_model()

    print('Is there any chance of rain: {:s}'.format(rain_predictor.get_rain_prediction(logistic, [[32, 98]])))
    print('The percentage chances calculated by the algorithm: {0:.2%}'
          .format(rain_predictor.get_rain_prediction_accuracy(logistic, [[32, 98]])))


def test_rain_naive_bayes():
    rain_predictor = RainPredictor.RainPredictor()
    # rain_predictor.forget_training()  # This will train the model on each execution
    if rain_predictor.get_model_trained_status() is False:
        print("\nTraining...")
        data_frame = rain_predictor.read_training_data_set(column_indices=[8, 10, 15])
        data_frame = rain_predictor.rename_columns(data_frame, ['isRain', 'temperature', 'humidity'])
        data_frame = rain_predictor.replace_data(data_frame, 'isRain', re.compile('^.*(RA|SN|DN|PL).*$'), 'Yes')
        data_frame = rain_predictor.replace_data(data_frame, 'isRain', re.compile('^(?!.*Yes).*$'), 'No')
        sns.countplot(y=data_frame['isRain'], data=data_frame)
        plt.show()
        (logistic, x_test, y_test) = rain_predictor.train_model_naive_bayes(
            data_frame[['temperature'] + ['humidity']], data_frame[['isRain']])
        print('Accuracy of naive bayes classifier on test set: {:.2f}'.format(logistic.score(x_test, y_test)))
        # Display the confusion matrix
        c_matrix = rain_predictor.get_confusion_matrix(logistic, x_test, y_test)
        rain_predictor.display_confusion_matrix(c_matrix)
    else:
        print("\nReusing trained data....")
        logistic = rain_predictor.load_trained_model()

    print('Is there any chance of rain: {:s}'.format(rain_predictor.get_rain_prediction(logistic, [[32, 98]])))
    print('The percentage chances calculated by the algorithm: {0:.2%}'
          .format(rain_predictor.get_rain_prediction_accuracy(logistic, [[32, 98]])))
