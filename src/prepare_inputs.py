import tensorflow as tf
from math import ceil

from file import load_file


def prepare_tensors():
    batch_tensor_list = []

    master_dict = load_file('dictionary')
    sequence_dict = load_file('sequences')
    sequence_data = sequence_dict["sequences"]
    offset_data = sequence_dict["offsets"]

    for sequence, offset in zip(sequence_data, offset_data):
        tensor_list = []
        curr_offset = 0.0
        for i, (element_s, element_o) in enumerate(zip(sequence, offset)):
            tensor_list.append(tf.Variable([
                master_dict[element_s]["coords"][0],
                master_dict[element_s]["coords"][1],
                element_o - curr_offset],
                tf.float32))
            curr_offset = element_o
        for trim in range(ceil(len(tensor_list)/128)):
            if (trim + 1) * 128 > len(tensor_list):
                batch_tensor_list.append(tf.stack(
                    tensor_list[len(tensor_list)-128 : len(tensor_list)]))
            else:
                batch_tensor_list.append(tf.stack(tensor_list[0 + (trim * 128) : 128 + (trim * 128)]))
    batch_tensor = tf.stack(batch_tensor_list)
    return batch_tensor
