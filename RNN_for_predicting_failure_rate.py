# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:13:58 2018

@author: 601327
"""

import gzip
import pickle

from itertools import cycle
import random

import numpy as np
import pandas as pd
import time
import os
#os.environ['KERAS_BACKEND']='tensorflow-gpu'

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#import matplotlib.pyplot as plt
from operator import add
#import time

#from itertools import cycle
import itertools
import matplotlib as plt

from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras import backend as K






def pickle_dump(df, filename):
    with gzip.GzipFile(filename, 'wb') as f:
        pickle.dump(df, f)

def pickle_load(filename):
    with gzip.GzipFile(filename, 'rb') as f:
        return pickle.load(f)



####################################################################
def model_GRU(units_1st_layer, input_shape, num_outputs, dropout_1st_layer, recurrent_dropout_1st_layer,
              metrics, units_2nd_layer=None, droput_2nd_layer=0.0, recurrent_dropout_2nd_layer=0.0):
    model = Sequential()

    if units_2nd_layer:
        model.add(layers.GRU(units_1st_layer,
                             dropout=dropout_1st_layer,
                             recurrent_dropout=recurrent_dropout_1st_layer,
                             return_sequences=True,
                             input_shape=input_shape))
        model.add(layers.GRU(units_2nd_layer, activation='relu',
                             dropout=droput_2nd_layer,
                             recurrent_dropout=recurrent_dropout_2nd_layer))
    else:
        model.add(layers.GRU(units_1st_layer,
                             dropout=dropout_1st_layer,
                             recurrent_dropout=recurrent_dropout_1st_layer,
                             input_shape=input_shape))
    model.add(layers.Dense(num_outputs, activation='sigmoid'))
    model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=metrics)

    return model
####################################################################

def generator(data, lookback, batch_size, feature_cols, target_cols,
              shuffle=True):

    data_gr = data.groupby(['id_student','module_presentation_identifier']).apply(lambda x: len(x))
    i = 0
    lst_min_max_indices = []
    len_df_lst = data_gr.values

    # Hier wird nur die letzte Order verwendet
    for c in range(len(len_df_lst)):
        num_orders = len_df_lst[c]
        ix_max = i + num_orders

        if ix_max - i >= lookback:
            ix_min = ix_max - lookback
        else:
            ix_min = i
        lst_min_max_indices.append([ix_min,ix_max])
        i+=num_orders

    if shuffle:
        random.seed(159)
        random.shuffle(lst_min_max_indices)


    steps = int(len(lst_min_max_indices) / batch_size)
    batches_min_max_indices = []

    for step in range(steps+1):
        batches_min_max_indices.append(lst_min_max_indices[step*batch_size:(step+1)*batch_size])
    if len(batches_min_max_indices[-1]) != batch_size:
        del batches_min_max_indices[-1]

    batches_cycle = cycle(batches_min_max_indices)

    data_features = data[feature_cols].values
    data_target = data[target_cols].values



    while 1:
        batches_indices = next(batches_cycle)

        samples_lst = []
        targets_lst = []
        for k, (x_min, x_max) in enumerate(batches_indices):
            data_sample = data_features[x_min:x_max]
            if data_sample.shape[0] < lookback:
                data_sample = np.append(np.zeros((lookback-data_sample.shape[0],data_sample.shape[-1])),
                        data_sample, axis=0)


            samples_lst.append(data_sample)
            targets_lst.append(data_target[x_max-1])

        samples = np.array(samples_lst)
        targets = np.array(targets_lst)
        yield samples, targets

##################################################################################

def model_fit_no_val(model, class_weights, train_gen, steps_per_epoch,
              epochs, verbose=1):

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  class_weight=class_weights,
                                  verbose=verbose)
    return history, model

##############################################################################

def f1_score_cl1_thres(threshold=0.5):
    def f1_score_cl1(y_true, y_pred_proba):

        # Runden der Wahrscheinlichkeiten --> bei einer Wahrscheinlichkeit über des Grenzwertes
        # threshold wird diese Klasse vorhergesagt
        y_pred = K.cast(K.greater(y_pred_proba, threshold), K.floatx())

        # Nur Labels der positiven Klasse
        y_true_cl1 = y_true[:,1]
        y_pred_cl1 = y_pred[:,1]

        # Berechnung der True Positives
        tmp = y_pred_cl1 + y_true_cl1
        tmp_bool = K.equal(tmp,2)
        tp = K.tf.count_nonzero(tmp_bool, dtype=y_true_cl1.dtype)

        # Berechnung der Anzahl vorhergesagten Samples mit Klasse 1
        pred = K.sum(y_pred_cl1)

        # Berechnung der Anzahl der tatsächlich Samples mit Klasse 1
        true = K.sum(y_true_cl1)

        # Berechnung Precision, Recall, f1-Score (Addition einer kleinen Zahl Epsilon, um Division durch 0 zu vermeiden)
        precision = tp/(pred + K.epsilon())
        recall = tp/(true + K.epsilon())

        f1 = 2 * (precision * recall)/(precision + recall + K.epsilon())
        return f1
    return f1_score_cl1

################################################################################

def calc_metric(model, data, gen, gen_steps, batch_size, threshold_lst):

    # Vorhersage für alle Daten mithilfe des trainierten Modells
    y_pred_proba_cl1_lst = []
    for i in range(gen_steps):
        X, y = next(gen)
        y_pred = model.predict(X)[:,1]
        y_pred_proba_cl1_lst.append(y_pred)

    y_pred_proba_cl1_lst = list(itertools.chain(*y_pred_proba_cl1_lst))
    # Datensätze, für die die Vorhersage gemacht wurde, inkl. Order-Id, damit pro Order der f1-Score
    # berechnet werden kann
    df_pred = data.groupby(['id_student','module_presentation_identifier']).last().reset_index().iloc[:gen_steps*batch_size]\
            [['module_presentation_identifier','final_result_binary_1']]

    del data

    mean_f1_score_lst = []
    for threshold in threshold_lst:
        y_pred_cl1_lst = [1 if y > threshold else 0 for y in y_pred_proba_cl1_lst]
        y_true_cl1_lst = df_pred['final_result_binary_1'].values.tolist()

        df_pred['reordered_1_pred'] = [1 if y > threshold else 0 for y in y_pred_cl1_lst]

        # Berechnung der Accuracy und der Konfusion-Matrix
        accuracy = accuracy_score(y_true_cl1_lst, y_pred_cl1_lst)
        conf_matrix = confusion_matrix(y_true_cl1_lst, y_pred_cl1_lst)
        print('conf_matrix_train')
        plot_confusion_matrix(conf_matrix)
        print('Accuracy: ', accuracy)
        print(classification_report(y_true_cl1_lst, y_pred_cl1_lst))


        # Berechnung der Precision, Recall und des f1-Scores pro Order
        # Bestimmung der True Positives
        df_pred['tp'] = [1 if x_sum==2 else 0 for x_sum in list(map(add, y_true_cl1_lst, y_pred_cl1_lst))]

        df_pred['reordered_1_pred'] = y_pred_cl1_lst

        # Summe über True Positives, Anzahl vorhergesagter und tatsächlich wieder Mittlerer f1-Scoreestellter Produkte pro Order-Id
        pred_order = df_pred.groupby(['module_presentation_identifier'])[['tp','final_result_binary_1','reordered_1_pred']].sum().values

        # Anpassung der True Positives --> wenn gar kein Produkt gekauft wurde und das auch vorhergesagt wurde, dann
        # ist das eine korrekte Vorhersage, True Positive wird auf 1 gesetzt, genauso die Anzahl vorhergesagter
        # und tatsächlicher Zustände (der Zustand "nichts wurde wiederbestellt" ist eine Vorhersage)

        # Dort wo pred_order überall 0 ist (nichts wurde gekauft und auch so vorhergesagt), wird die Zeile
        # durch [1,1,1] ersetzt
        arr_default = np.ones_like(pred_order)
        cond_1 = (pred_order == 0).all(axis=1)
        pred_order[cond_1] = arr_default[cond_1]

        # Dort wo vorhergesagt wurde, dass nichts gekauft wird und dort wo tatsächlich nichts gekauft wird
        # wird die Anzahl der Vorhersagen auf 1 korrigiert, d.h. eine 0 in Spalte 1 und 2 wird durch 1 ersetzt
        cond_2 = pred_order[:,1] == 0
        cond_3 = pred_order[:,2] == 0
        pred_order[:,1][cond_2] = 1
        pred_order[:,2][cond_3] = 1

        # Berechnung von Precision und Recall

        prec = pred_order[:,0] / pred_order[:,2]
        recall = pred_order[:,0] / pred_order[:,1]

        # Berechnung f1-Score
        f1 = np.divide(2*prec*recall, prec+recall, out=np.zeros_like(prec), where=(prec+recall)!=0)
        mean_f1_score_lst.append(f1.mean())

    print('Mittlerer f1-Score pro Order:', mean_f1_score_lst)
    return mean_f1_score_lst, accuracy, y_pred_cl1_lst, y_true_cl1_lst, X


################################################################################

def mean_f1_score_per_order(model, data, gen, gen_steps, batch_size, threshold_lst):
    # Vorhersage für alle Daten mithilfe des trainierten Modells
    y_pred_proba_cl1_lst = []
    for i in range(gen_steps):
        X, y = next(gen)
        y_pred = model.predict(X)[:,1]
        y_pred_proba_cl1_lst.append(y_pred)

    y_pred_proba_cl1_lst = list(itertools.chain(*y_pred_proba_cl1_lst))
    # Datensätze, für die die Vorhersage gemacht wurde, inkl. Order-Id, damit pro Order der f1-Score
    # berechnet werden kann
    df_pred = data.groupby(['id_student','module_presentation_identifier']).last().reset_index().iloc[:gen_steps*batch_size]\
            [['module_presentation_identifier','final_result_binary_1']]

    del data
    mean_f1_score_lst = []
    for threshold in threshold_lst:
        y_pred_cl1_lst = [1 if y > threshold else 0 for y in y_pred_proba_cl1_lst]
        y_true_cl1_lst = df_pred['final_result_binary_1'].values.tolist()


    # Berechnung der Precision, Recall und des f1-Scores pro Order
    # Bestimmung der True Positives
    df_pred['tp'] = [1 if x_sum==2 else 0 for x_sum in list(map(add, y_true_cl1_lst, y_pred_cl1_lst))]

    df_pred['reordered_1_pred'] = y_pred_cl1_lst

    # Summe über True Positives, Anzahl vorhergesagter und tatsächlich wieder bestellter Produkte pro Order-Id
    pred_order = df_pred.groupby(['module_presentation_identifier'])[['tp','final_result_binary_1','reordered_1_pred']].sum().values

    # Anpassung der True Positives --> wenn gar kein Produkt gekauft wurde und das auch vorhergesagt wurde, dann
    # ist das eine korrekte Vorhersage, True Positive wird auf 1 gesetzt, genauso die Anzahl vorhergesagter
    # und tatsächlicher Zustände (der Zustand "nichts wurde wiederbestellt" ist eine Vorhersage)
    # Dort wo pred_order überall 0 ist (nichts wurde gekauft und auch so vorhergesagt), wird die Zeile
    # durch [1,1,1] ersetzt
    arr_default = np.ones_like(pred_order)
    cond_1 = (pred_order == 0).all(axis=1)
    pred_order[cond_1] = arr_default[cond_1]

    # Dort wo vorhergesagt wurde, dass nichts gekauft wird und dort wo tatsächlich nichts gekauft wird
    # wird die Anzahl der Vorhersagen auf 1 korrigiert, d.h. eine 0 in Spalte 1 und 2 wird durch 1 ersetzt
    cond_2 = pred_order[:,1] == 0
    cond_3 = pred_order[:,2] == 0
    pred_order[:,1][cond_2] = 1
    pred_order[:,2][cond_3] = 1
   # Berechnung von Precision und Recall
    prec = pred_order[:,0] / pred_order[:,2]
    recall = pred_order[:,0] / pred_order[:,1]
    # Berechnung f1-Score
    f1 = np.divide(2*prec*recall, prec+recall, out=np.zeros_like(prec), where=(prec+recall)!=0)
    mean_f1_score_lst.append(f1.mean())
    del pred_order
    del prec
    del recall
    del df_pred
    print(mean_f1_score_lst)
    print(y_pred_cl1_lst)
    return mean_f1_score_lst, y_pred_cl1_lst

########################################################################################

def plot_confusion_matrix(confmat):
    fig, ax = plt.pyplot.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,
            s=confmat[i, j],va='center', ha='center')
    plt.pyplot.xlabel('predicted label')
    plt.pyplot.ylabel('true label')

#########################################################################################

df_train = pd.read_pickle('../df_train.pkl')
df_test_1 = pd.read_pickle('../df_test_1.pkl')

sample_size = 'all_train'
filename_valid = "output.csv"


epochs = 5

lookback = 240

# 1. Layer
units_1st_layer = 8
dropout_1st_layer = 0.2
recurrent_dropout_1st_layer = 0.1

# 2. Layer
units_2nd_layer = None
dropout_2nd_layer = None
recurrent_dropout_2nd_layer = None



batch_size = 1024
class_weights = {0:7245, 1:4017}

only_last_order = True
THRES_lst = [0.5]


numericFeaturesCols = [
        'date',
        'sum_click',
        'studied_credits',
        'num_of_prev_attempts'
        ]

feature_cols = [
        'code_module_AAA',
        'code_module_BBB',
       'code_module_CCC',
       'code_module_DDD',
       'code_module_EEE',
       'code_module_FFF',
       'code_module_GGG',
       'code_presentation_2013B',
       'code_presentation_2013J',
       'code_presentation_2014B',
       'code_presentation_2014J',
       'gender_F',
       'gender_M',
       'region_East Anglian Region',
       'region_East Midlands Region',
       'region_Ireland',
       'region_London Region',
       'region_North Region',
       'region_North Western Region',
       'region_Scotland',
       'region_South East Region',
       'region_South Region',
       'region_South West Region',
       'region_Wales',
       'region_West Midlands Region',
       'region_Yorkshire Region',
       'highest_education_A Level or Equivalent',
       'highest_education_HE Qualification',
       'highest_education_Lower Than A Level',
       'highest_education_No Formal quals',
       'highest_education_Post Graduate Qualification',
       'imd_band_0-10%',
       'imd_band_10-20',
       'imd_band_20-30%',
       'imd_band_30-40%',
       'imd_band_40-50%',
       'imd_band_50-60%',
       'imd_band_60-70%',
       'imd_band_70-80%',
       'imd_band_80-90%',
       'imd_band_90-100%',
       'age_band_0-35',
       'age_band_35-55',
       'age_band_55<=',
       'disability_N',
       'disability_Y'
 ]
target_cols = ['final_result_binary_0', 'final_result_binary_1']



input_shape_cols = len(feature_cols)

model = model_GRU(units_1st_layer,
                  input_shape=(None, input_shape_cols),
                  num_outputs=len(target_cols),
                  dropout_1st_layer=dropout_1st_layer,
                  recurrent_dropout_1st_layer=recurrent_dropout_1st_layer,
                  metrics=[f1_score_cl1_thres(0.5)],
                  units_2nd_layer=units_2nd_layer,
                  droput_2nd_layer=dropout_2nd_layer,
                  recurrent_dropout_2nd_layer=recurrent_dropout_2nd_layer)




# Normierungsparameter
#mean, std = utilsAL.get_mean_std(filenames_train_model_lst, numericFeaturesCols)

norm_params_df = pd.DataFrame(columns=numericFeaturesCols, index=['mean','std'])
for i in numericFeaturesCols:
    norm_params_df[str(i)]['mean'] = df_train[str(i)].mean()
    norm_params_df[str(i)]['std'] = df_train[str(i)].std()

#norm_params_df.to_csv("data/norm_params_"+sample_size+".csv")

#norm_params_df = pd.read_csv("data/norm_params_"+sample_size+".csv", index_col=0)
mean = norm_params_df.loc['mean'].values
std = norm_params_df.loc['std'].values



df_results = pd.DataFrame()

#Training
timestamp = time.localtime()
best_f1_score_lst = []
count_worse_score = 0
count_not_better = 0

train_f1_list = []
train_accuracy = []
test_f1_list = []
test_accuracy = []
score_list = []


for epoch in range(1, epochs+1):
    start = time.time()

    data_train = df_train


    # Standardisierung
    data_train.loc[:,numericFeaturesCols] -= mean
    data_train.loc[:,numericFeaturesCols] /= std

    # Generator initialisieren
    steps_per_epoch = int(len(data_train.groupby(['id_student','module_presentation_identifier'])\
                              [['id_student','module_presentation_identifier']].last()) / batch_size)
    print("train_steps = " + str(steps_per_epoch))

    train_gen = generator(data_train,
                                  lookback,
                                  batch_size,
                                  feature_cols,
                                  target_cols,
                                  shuffle=True)

    # Modell weitertrainieren
    history, model = model_fit_no_val(model,
                              class_weights,
                              train_gen,
                              steps_per_epoch,
                              1)

    train_gen = generator(data_train,
                                  lookback,
                                  batch_size,
                                  feature_cols,
                                  target_cols,
                                  shuffle=False,
                                  )

    print('calc_metric - data_train')
    mean_f1_score_lst, accuracy, y_pred_cl1_lst_train ,y_true_cl1_lst_train, X_train = calc_metric(model, data_train, train_gen, steps_per_epoch, batch_size, THRES_lst)

    train_f1_list.append(mean_f1_score_lst)
    train_accuracy.append(accuracy)
    del data_train

    # Ende einer Epoche: das Modell wird zwischengespeichert und validiert


    ###############################################################################

    end = time.time()
    print('Durchgang Training Epoche %i: %f'%(epoch,end-start))
    # Test
    start = time.time()
    # Einlesen der Daten
    data_test = df_test_1

    # Standardisierung

    data_test.loc[:,numericFeaturesCols] -= mean
    data_test.loc[:,numericFeaturesCols] /= std

    # Generator initialisieren
    test_steps = int(len(data_test.groupby(['id_student','module_presentation_identifier'])\
                              [['id_student','module_presentation_identifier']].last()) / batch_size)
    print("test_steps = " + str(test_steps))
    test_gen = generator(data_test,
                         lookback,
                         batch_size,
                         feature_cols,
                         target_cols,
                         shuffle=False)



    # Speichern der Evaluierung des Trainings


    print('calc_metric - data_test')

    mean_f1_score_lst, accuracy, y_pred_cl1_lst_test, y_true_cl1_lst_test, X_test = calc_metric(model, data_test, test_gen, test_steps, batch_size, THRES_lst)


    test_f1_list.append(mean_f1_score_lst)
    test_accuracy.append(accuracy)

    del data_test

    #if c == 0:
#    mean_f1_score_per_order_arr = [mean_f1_score_per_order_part]
    #else:
    #    mean_f1_score_per_order_arr = np.append(mean_f1_score_per_order_arr, [mean_f1_score_per_order_part], axis=0)


#    mean_f1_score_per_order = np.mean(mean_f1_score_per_order_arr, axis=0)


    end = time.time()
    print('Epoche Test %i: %f'%(epoch,end-start))
    print(mean_f1_score_per_order)


    # Test, ob Training abgebrochen werden soll
    best_f1_score = np.max(mean_f1_score_lst)
    print('Bester Score:',best_f1_score)
    score_list.append(best_f1_score)

    best_f1_score_lst.append(best_f1_score)
    best_f1_score_prior = np.max(best_f1_score_lst)
    if best_f1_score < (best_f1_score_prior-0.001):
        count_worse_score += 1
    if best_f1_score < (best_f1_score_prior):
        count_not_better += 1
    if count_worse_score >= 3:
        print('Abbruch: bei Epoche %i wurde das beste Ergebnis 3 mal unterboten'%epoch)
#        break
    if count_not_better >=5:
        print('Abbruch: bei Epoche %i wurde das beste Ergebnis 5 mal nicht verbessert'%epoch)
#        break

df_results[('BATCH_SIZE' + str(batch_size))] = best_f1_score_lst

