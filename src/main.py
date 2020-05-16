# MIDI files copyright holder
# Name: Bernd Krueger
# Source: http://www.piano-midi.de


import tensorflow as tf
import numpy as np
from os import listdir

from encode_data import prepare_data
from w2v import build_word2vec_model
from prep_learn_data import prepare_tensors
from decode_data import prepare_output
from utilities import *
from file import load_file


if __name__ == "__main__":
    # prepare_data for : chopin / beeth / tschai
    if prompt_question("Build new input data dictionaries?"):
        prepare_data('..\MIDIs\\chopin\\')
        build_word2vec_model()
    else:
        print('skipping...')
        try:
            load_file('dictionary')
            load_file('sequences')
            load_file('w2v_vocab')
        except FileNotFoundError:
            print("Input data dictionaries not found. Building new dictionaries.")

    if prompt_question("Build new tensors?"):
        input_data, target_data = prepare_tensors(length_of_vector=128, partition=1)
    else:
        print('skipping...')
        try:
            input_data = load_file('input_tensor')
            target_data = load_file('target_tensor')
        except FileNotFoundError:
            print("Tensors not found. Building new tensors.")
            input_data, target_data = prepare_tensors(length_of_vector=128, partition=1)

    if prompt_question("Build new model?"):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(3, input_shape=(128, 3), return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(3, tf.keras.layers.Activation('relu')))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(input_data, target_data, epochs=10, verbose=1)

        if prompt_question("Save model?"):
            model.save('..\models\model_' + gen_timestamp())

        if prompt_question("Build sequence?"):
            noise_vec = gen_noise_vector(batch_size=1, time_steps=128)
            iter_num = 0
            for iteration in range(1):
                noise_vec = model.predict(noise_vec)
                iter_num = iteration
            prediction = np.reshape(noise_vec, (128, 3))
            prepare_output(prediction, iter_num)
    else:
        print('skipping...')

    if prompt_question("Build sequences on known models?"):
        noise_vec = gen_noise_vector(batch_size=1, time_steps=128)
        prepare_output(np.reshape(noise_vec, (128, 3)), 0, "noise")
        vocab = load_file("w2v_vocab")
        coords = np.asarray(vocab["coords"])
        coords = np.asarray(coords)
        for directory in listdir('..\models'):
            print("processing: %s..." % directory)
            model = tf.keras.models.load_model('../models/' + directory)
            curr_noise = noise_vec
            for iteration in range(100):
                curr_noise = model.predict(curr_noise)
                curr_noise = rectify_vector(curr_noise, coords)
                if iteration == 0 or iteration == 9 or iteration == 24 or iteration == 99:
                    prediction = np.reshape(curr_noise, (128, 3))
                    prepare_output(prediction, iteration, directory)

    else:
        print('skipping...')


