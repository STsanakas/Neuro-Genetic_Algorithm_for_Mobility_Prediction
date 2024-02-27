import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from time import time
from tabulate import tabulate
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout
from mobility import cartesian_error
import math

def jac(a, b):
    if a==b:
       return 1
    else:
       return 0

def clearnan(x):
    if math.isnan(x):
        return 10000000
    return x

def similarity (best, model1, model2, a=0.35, b=0.15, c=0.5):      
    st1=[model1['parameters']['lstms'], model1['parameters']['denses']]
    st2=[model2['parameters']['lstms'], model2['parameters']['denses']]
    tr1=[model1['parameters']['activation1'], model1['parameters']['activation2'], model1['parameters']['activation3'],model1['parameters']['activation4'],model1['parameters']['activation5'],model1['parameters']['optimizer']]
    tr2=[model2['parameters']['activation1'], model2['parameters']['activation2'], model2['parameters']['activation3'],model2['parameters']['activation4'],model2['parameters']['activation5'],model1['parameters']['optimizer']]
    nom=0
    for i in range(len(st1)):
       nom+=st1[i]*st2[i]
    den=math.sqrt(sum(i ** 2 for i in st1))*math.sqrt(sum(i ** 2 for i in st2))
    struct=nom/den
    train=0
    for i in range(len(tr1)):
        train+=((jac(tr1[i], tr2[i]))/len(tr1))        
    model1['error'] = clearnan(model1['error'])
    model2['error'] = clearnan(model2['error'])    
    acc=((best/model1['error'])+(best/model2['error']))/2
    sim=a*struct+b*train+c*acc
    return sim


def print_gen_stats(DL_model, genetic_stats):
    plt.title('Best model train and validation loss during training')
    plt.ylabel('error (m)')
    plt.xlabel('epoch')
    plt.plot(DL_model['history1'].history['loss'] +
             DL_model['history2'].history['loss'])
    plt.plot(DL_model['history1'].history['val_loss'] +
             DL_model['history2'].history['val_loss'])
    plt.legend(['train loss', 'validation loss'], loc='upper right')
    plt.show()
    plt.title('Best model over time')
    plt.ylabel('best model error (m)')
    plt.xlabel('time')
    plt.plot(np.array(genetic_stats['best_time'])[
             :, 1], np.array(genetic_stats['best_time'])[:, 0])
    plt.show()
    plt.title('Best and average error over genetic iterations')
    plt.ylabel('error (m)')
    plt.xlabel('iteration')
    plt.plot(genetic_stats['iteration_best'])
    plt.plot(genetic_stats['iteration_average'])
    plt.legend(['iteration best', 'iteration average'], loc='upper right')
    plt.show()
    return


def get_optimizer(optimizer, lr):
    optimizers = {
        'adam': tf.keras.optimizers.Adam(learning_rate=lr),
        'adamax': tf.keras.optimizers.Adamax(learning_rate=lr),
        'sgd': tf.keras.optimizers.SGD(learning_rate=lr),
        'nadam': tf.keras.optimizers.Nadam(learning_rate=lr),
        'adadelta': tf.keras.optimizers.Adadelta(learning_rate=lr),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=lr),
        'adagrad': tf.keras.optimizers.Adagrad(learning_rate=lr),
        'ftrl': tf.keras.optimizers.Ftrl(learning_rate=lr)}
    return optimizers.get(optimizer)


def build_model(nn_params, trainX, trainY, testX, testY, train_size, test_size, iter, models_path):
    global best, best_ones, start_time, hb_threshold
    network = Sequential()
    if nn_params['lstms'] == 2:
        network.add(
            LSTM(nn_params['lstm1'], input_shape=trainX[0].shape, return_sequences=True))
        network.add(LSTM(nn_params['lstm2']))
    elif nn_params['lstms'] == 1:
        network.add(LSTM(nn_params['lstm1'], input_shape=trainX[0].shape))
    network.add(Flatten())

    network.add(Dense(nn_params['dense1'],
                activation=nn_params['activation1']))
    network.add(Dropout(nn_params['dropout1']))
    if nn_params['denses'] >= 2:
        network.add(Dense(nn_params['dense2'],
                    activation=nn_params['activation2']))
        network.add(Dropout(nn_params['dropout2']))
        if nn_params['denses'] >= 3:
            network.add(Dense(nn_params['dense3'],
                        activation=nn_params['activation3']))
            network.add(Dropout(nn_params['dropout3']))
            if nn_params['denses'] >= 4:
                network.add(
                    Dense(nn_params['dense4'], activation=nn_params['activation4']))
                network.add(Dropout(nn_params['dropout4']))
                if nn_params['denses'] >= 5:
                    network.add(
                        Dense(nn_params['dense5'], activation=nn_params['activation5']))
                    network.add(Dropout(nn_params['dropout5']))
    network.add(Dense(trainY[0].shape[0], activation='softmax'))
    network.compile(loss=cartesian_error, optimizer=get_optimizer(
        nn_params['optimizer'], nn_params['learning_rate']))
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=min(
        max(2*iter, 3), 15), min_delta=0.0001, restore_best_weights=True)]
    history1 = network.fit(trainX[:train_size], trainY[:train_size], epochs=15,
                           batch_size=nn_params['batch_size'], verbose=0, validation_split=0.2, callbacks=callback)
    error_p1 = network.evaluate(testX[:test_size], testY[:test_size], verbose=0)
    error = error_p1
    history2 = None
    if error_p1<hb_threshold:
        history2 = network.fit(trainX[:train_size], trainY[:train_size], epochs=150,
                               batch_size=nn_params['batch_size'], verbose=0, validation_split=0.2, callbacks=callback)
        error = network.evaluate(testX[:test_size], testY[:test_size], verbose=0)        

    if error < best:
        best = error
        best_ones.append((error, int(time()-start_time)))
        network.save(models_path+'genetic_classification.h5')
    model = {}
    model['history1'] = history1
    model['history2'] = history2
    model['error'] = error
    # model['model']=network
    del network
    model['error_p1'] = error_p1    
    model['parameters'] = nn_params
    model['when'] = int(time()-start_time)
    model['iteration'] = iter
    return model


def create_random_network():
    params = {}
    params['lstms'] = random.choice((1, 2))
    params['denses'] = random.choice((1, 2, 3, 4, 5))
    params['lstm1'] = random.choice((4, 8, 16, 32, 64, 128, 256, 512))
    params['lstm2'] = random.choice((4, 8, 16, 32, 64, 128, 256, 512))
    params['dense1'] = random.choice((4, 8, 16, 32, 64, 128, 256, 512))
    params['dense2'] = random.choice((4, 8, 16, 32, 64, 128, 256, 512))
    params['dense3'] = random.choice((4, 8, 16, 32, 64, 128, 256, 512))
    params['dense4'] = random.choice((4, 8, 16, 32, 64, 128, 256, 512))
    params['dense5'] = random.choice((4, 8, 16, 32, 64, 128, 256, 512))
    params['dropout1'] = random.choice((0, 0.1, 0.2, 0.3, 0.4, 0.5))
    params['dropout2'] = random.choice((0, 0.1, 0.2, 0.3, 0.4, 0.5))
    params['dropout3'] = random.choice((0, 0.1, 0.2, 0.3, 0.4, 0.5))
    params['dropout4'] = random.choice((0, 0.1, 0.2, 0.3, 0.4, 0.5))
    params['dropout5'] = random.choice((0, 0.1, 0.2, 0.3, 0.4, 0.5))
    params['activation1'] = random.choice(
        ('relu', 'elu', 'sigmoid', 'tanh', 'linear'))
    params['activation2'] = random.choice(
        ('relu', 'elu', 'sigmoid', 'tanh', 'linear'))
    params['activation3'] = random.choice(
        ('relu', 'elu', 'sigmoid', 'tanh', 'linear'))
    params['activation4'] = random.choice(
        ('relu', 'elu', 'sigmoid', 'tanh', 'linear'))
    params['activation5'] = random.choice(
        ('relu', 'elu', 'sigmoid', 'tanh', 'linear'))
    params['optimizer'] = random.choice(
        ('adam', 'adamax', 'adagrad', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'ftrl'))
    params['learning_rate'] = random.choice(
        (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1))
    params['batch_size'] = random.choice((16, 32, 64, 128, 256, 512, 1024))
    return params


def mate(model1, model2, chance=0.5):
    probability = [1-chance, chance]
    params = {}
    params['lstms'] = np.random.choice(
        (model1['lstms'], model2['lstms']), p=probability)
    params['denses'] = np.random.choice(
        (model1['denses'], model2['denses']), p=probability)
    params['lstm1'] = np.random.choice(
        (model1['lstm1'], model2['lstm1']), p=probability)
    params['lstm2'] = np.random.choice(
        (model1['lstm2'], model2['lstm2']), p=probability)
    params['dense1'] = np.random.choice(
        (model1['dense1'], model2['dense1']), p=probability)
    params['dense2'] = np.random.choice(
        (model1['dense2'], model2['dense2']), p=probability)
    params['dense3'] = np.random.choice(
        (model1['dense3'], model2['dense3']), p=probability)
    params['dense4'] = np.random.choice(
        (model1['dense4'], model2['dense4']), p=probability)
    params['dense5'] = np.random.choice(
        (model1['dense5'], model2['dense5']), p=probability)
    params['dropout1'] = np.random.choice(
        (model1['dropout1'], model2['dropout1']), p=probability)
    params['dropout2'] = np.random.choice(
        (model1['dropout2'], model2['dropout2']), p=probability)
    params['dropout3'] = np.random.choice(
        (model1['dropout3'], model2['dropout3']), p=probability)
    params['dropout4'] = np.random.choice(
        (model1['dropout4'], model2['dropout4']), p=probability)
    params['dropout5'] = np.random.choice(
        (model1['dropout5'], model2['dropout5']), p=probability)
    params['activation1'] = np.random.choice(
        (model1['activation1'], model2['activation1']), p=probability)
    params['activation2'] = np.random.choice(
        (model1['activation2'], model2['activation2']), p=probability)
    params['activation3'] = np.random.choice(
        (model1['activation3'], model2['activation3']), p=probability)
    params['activation4'] = np.random.choice(
        (model1['activation4'], model2['activation4']), p=probability)
    params['activation5'] = np.random.choice(
        (model1['activation5'], model2['activation5']), p=probability)
    params['optimizer'] = np.random.choice(
        (model1['optimizer'], model2['optimizer']), p=probability)
    params['learning_rate'] = np.random.choice(
        (model1['learning_rate'], model2['learning_rate']), p=probability)
    params['batch_size'] = np.random.choice(
        (model1['batch_size'], model2['batch_size']), p=probability)
    return params


def genetic(trainX, trainY, testX, testY, train_size, test_size, models_path, iters=10, population=20, survival_rate=0.4, mutation_chance=0.1):
    global best, best_ones, start_time, hb_threshold    
    start_time = time()
    best_ones = []
    best = 0
    hb_threshold = 9999999
    print('GeneticDL training on', train_size, 'samples, testing on',
          test_size, 'samples for', iters, 'iterations with population', population)
    models = []
    iteration_best = []
    iteration_average = []
    for i in range(iters):
        if i == 0:
            for j in tqdm(range(population)):
                models.append(build_model(create_random_network(
                ), trainX, trainY, testX, testY, train_size, test_size, i, models_path))
        else:
            del models
            models = next_models
            #####################################
            combinations=[]
            similarities=[]
            for num1 in range(int(population*survival_rate)):
                for num2 in range(int(population*survival_rate)):
                    if num1<num2:
                        combinations.append((prev_models[num1], prev_models[num2]))
                        similarities.append(similarity(best, prev_models[num1], prev_models[num2]))
            similarities=[number ** 30 for number in similarities]
            similarities=[number/sum(similarities) for number in similarities]
            #####################################
            for j in tqdm(range(len(models), population)):
                parents=random.choices(combinations, weights=similarities)     
                child = mate(parents[0][0]['parameters'],
                             parents[0][1]['parameters'])
                mutated_child = mate(
                    child, create_random_network(), mutation_chance)
                model = build_model(mutated_child, trainX, trainY,
                                    testX, testY, train_size, test_size, i, models_path)
                models.append(model)
                del model
            del next_models
        models = sorted(models, key=lambda i: i['error'], reverse=False)
        next_models = []
        ###########################
        prev_models = models.copy()
        ###########################
        for j in range(int(survival_rate*population)):
            next_models.append(models[j])
        # print(p1_target, 'is the next p1 target')
        iteration_best.append(models[0]['error'])
        av = 0
        for m in next_models:
            av += m['error_p1']
        iteration_average.append(av/int(population*survival_rate))
        hb_threshold = av/int(population*survival_rate)
        results = np.zeros((population, 4))
        for res in range(population):
            results[res, 0] = models[res]['error']
            results[res, 1] = models[res]['error_p1']
            results[res, 2] = models[res]['when']
            results[res, 3] = models[res]['iteration']+1
        print(tabulate(pd.DataFrame(results, columns=['error(m)', 'phase_1 error', 'time to discover', 'generation'], index=list(
            map(str, range(1, population+1)))), headers='keys', tablefmt='psql'))
    best_ones.append((models[0]['error'], int(time()-start_time)))
    print('GeneticDL minimum error: ',
          models[0]['error'], 'in ', int(time()-start_time), 's')
    genetic_stats = {'iteration_best': iteration_best,
                     'iteration_average': iteration_average, 'best_time': best_ones}
    return (models, genetic_stats)
