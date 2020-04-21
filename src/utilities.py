import numpy as np
import random as rn
import datetime as dt

from file import load_file


def gen_noise_vector(batch_size=1, time_steps=128):
    vocab = load_file("w2v_vocab")
    coords = np.asarray(vocab["coords"])
    test_pred_data = []

    for no_of_vectors in range(batch_size):
        temp_test_pred_data = []
        for no_of_notes in range(time_steps):
            temp_test_pred_data.append([rn.uniform(np.argmin(coords[0]), np.argmax(coords[0])),
                                   rn.uniform(np.argmin(coords[1]), np.argmax(coords[1])),
                                   rn.uniform(0, 1)])
        test_pred_data.append(temp_test_pred_data)
    return np.array(test_pred_data)


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