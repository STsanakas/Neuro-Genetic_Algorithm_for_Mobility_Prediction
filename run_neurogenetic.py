from hypertuning import genetic
from geolife_data_utilities import get_data, get_geolife_path
trainX, testX, trainY, testY = get_data('NE_walk')
genetic(trainX, trainY, testX, testY, int(0.1*trainX.shape[0]), int(0.1*testX.shape[0]), get_geolife_path()+'Geolife_models_test270224', iters=10, population=20, survival_rate=0.4, mutation_chance=0.1)