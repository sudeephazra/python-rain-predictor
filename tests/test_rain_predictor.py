import RainPredictor


def test_rain_predictor_file_read():
    rain_predictor = RainPredictor.RainPredictor('../data/Weather Dataset_Filtered.csv')
    dataframe = rain_predictor.read_file(columns=[8, 10, 15])
    print(dataframe.shape)