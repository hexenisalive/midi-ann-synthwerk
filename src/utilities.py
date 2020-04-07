import numpy as np
import random as rn
import datetime as dt

from file import load_file


def gen_noise_vector():
    vocab = load_file("w2v_vocab")
    coords = np.asarray(vocab["coords"])
    test_pred_data = []
    for no_of_notes in range(128):
        test_pred_data.append([rn.uniform(np.argmin(coords[0]), np.argmax(coords[0])),
                               rn.uniform(np.argmin(coords[1]), np.argmax(coords[1])),
                               rn.uniform(0, 1)])
    return np.array(test_pred_data)


def prompt_question(question):
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}

    while True:
        print(question + " [y/n] ")
        choice = input().lower()
        if choice in valid:
            return valid[choice]


def gen_timestamp():
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")