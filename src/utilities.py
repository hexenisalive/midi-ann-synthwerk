import numpy as np
import random as rn
import datetime as dt
import tensorflow as tf

from file import load_file


def gen_noise_vector(batch_size=1, time_steps=128, length=128, offset=0):
    vocab = load_file("w2v_vocab")
    coords = np.asarray(vocab["coords"])
    test_pred_data = []
    if time_steps-length-offset >= 0:
        for no_of_vectors in range(batch_size):
            temp_test_pred_data = []

            for no_of_notes_off in range(offset):
                temp_test_pred_data.append([0, 0, rn.uniform(0, 1)])

            for no_of_notes in range(length):
                temp_test_pred_data.append([rn.uniform(np.argmin(coords[0]), np.argmax(coords[0])),
                                       rn.uniform(np.argmin(coords[1]), np.argmax(coords[1])),
                                       rn.uniform(0, 1)])

            for no_of_notes_off in range(time_steps-length-offset):
                temp_test_pred_data.append([0, 0, rn.uniform(0, 1)])

            test_pred_data.append(temp_test_pred_data)
    return np.array(test_pred_data)


def gen_partial_vector(batch_size=1, time_steps=128, length=128, offset=0):

    master_dict = load_file('dictionary')
    sequence_dict = load_file('sequences')
    sequence_data = sequence_dict["sequences"]
    offset_data = sequence_dict["offsets"]
    test_pred_data = []

    if time_steps - length - offset >= 0:
        for no_of_vectors in range(batch_size):
            temp_test_pred_data = []
            partial_vector_index = int(rn.uniform(0, len(sequence_data)))
            partial_vector = sequence_data[partial_vector_index][:length]
            partial_offset_vector = offset_data[partial_vector_index][:length]

            for no_of_notes_off in range(offset):
                temp_test_pred_data.append([0, 0, rn.uniform(0, 1)])

            curr_offset = partial_offset_vector[0]
            for element_s, element_o in zip(partial_vector, partial_offset_vector):
                temp_test_pred_data.append([
                    master_dict[element_s]["coords"][0],
                    master_dict[element_s]["coords"][1],
                    element_o - curr_offset])
                curr_offset = element_o

            for no_of_notes_off in range(time_steps-length-offset):
                temp_test_pred_data.append([0, 0, rn.uniform(0, 1)])

            test_pred_data.append(temp_test_pred_data)
    return np.array(test_pred_data)


def gen_zero_vector(batch_size=1, time_steps=128):
    test_pred_data = []

    for no_of_vectors in range(batch_size):
        temp_test_pred_data = []

        for no_of_notes in range(time_steps):
            temp_test_pred_data.append([0, 0, rn.uniform(0, 1)])

        test_pred_data.append(temp_test_pred_data)
    return np.array(test_pred_data)


def rectify_vector(vector, coords):
    vector = np.reshape(vector, (128, 3))
    new_vector = vector
    for iteration, element in enumerate(vector):
        coord = [element[0], element[1]]
        dist = np.sum((coords - coord) ** 2, axis=1)
        new_vector[iteration][0] = coords[int(np.argmin(dist))][0]
        new_vector[iteration][1] = coords[int(np.argmin(dist))][1]

    return np.reshape(new_vector, (1, 128, 3))


def prompt_question(question):
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}

    print(question + " [y/n] ")
    while True:
        choice = input().lower()
        if choice in valid:
            return valid[choice]


def gen_timestamp():
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
