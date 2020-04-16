# MIDI files copyright holder
# Name: Bernd Krueger
# Source: http://www.piano-midi.de


import tensorflow as tf
import numpy as np

from encode_data import prepare_data
from w2v import build_word2vec_model
from prep_learn_data import prepare_tensors
from decode_data import prepare_output
from utilities import prompt_question, gen_noise_vector, gen_timestamp


if __name__ == "__main__":
    # prepare_data for : chopin / beeth / tschai
    if prompt_question("Build new input data dictionaries? (Make sure files 'dictionary.pkl', "
                       "'sequences.pkl', 'w2v_vocab.pkl' exist in directory './pkl_files'.)"):
        prepare_data('..\MIDIs\\chopin\\')
        build_word2vec_model()

    input_data, target_data = prepare_tensors(length_of_vector=128, partition=1)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.CuDNNLSTM(3, input_shape=(128, 3), return_sequences=True))
    model.add(tf.keras.layers.Dense(3))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(input_data, target_data, epochs=100, verbose=1)

    if prompt_question("Save model?"):
        model.save('..\models\model_' + gen_timestamp())

    if prompt_question("Build sequence?"):
        noise_vec = gen_noise_vector(batch_size=1, time_steps=128)
        for iteration in range(1):
            noise_vec = model.predict(noise_vec)
        prediction = np.reshape(noise_vec, (128, 3))
        prepare_output(prediction)



