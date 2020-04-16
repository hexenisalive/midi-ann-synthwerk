import tensorflow as tf
from math import ceil

from file import load_file


def prepare_tensors(length_of_vector=128, partition=128):
    batch_tensor_input = []
    batch_tensor_target = []

    master_dict = load_file('dictionary')
    sequence_dict = load_file('sequences')
    sequence_data = sequence_dict["sequences"]
    offset_data = sequence_dict["offsets"]

    print("Preparing sequences...")
    max_iterator = len(sequence_data)
    for iterator, (sequence, offset) in enumerate(zip(sequence_data, offset_data)):
        print("Processing sequence:", iterator+1, "/", max_iterator)
        tensor_list = []
        div_list = []
        curr_offset = 0.0
        for element_s, element_o in zip(sequence, offset):
            tensor_list.append(tf.Variable([
                master_dict[element_s]["coords"][0],
                master_dict[element_s]["coords"][1],
                element_o - curr_offset],
                tf.float32))
            curr_offset = element_o

        div_list = div_tensor_list(tensor_list, length_of_vector)

        batch_tensor_input += div_list[:-1:partition]
        batch_tensor_target += div_list[1::partition]

    print("Building tensors...")
    input_data = tf.stack(batch_tensor_input)
    target_data = tf.stack(batch_tensor_target)
    print("Done... shape:", input_data.shape)

    return input_data, target_data


def div_tensor_list(tensor_list, length_of_vector):
    div_list = []
    for iterator in range(len(tensor_list) - length_of_vector - 1):
        div_list.append(tensor_list[iterator: iterator+length_of_vector])
    return div_list
