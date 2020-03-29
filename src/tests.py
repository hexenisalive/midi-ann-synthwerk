from random import uniform

from file import stream_to_file
from decode_data import data_to_stream


def test_stream_to_file():
    # randomly generated simulated data for testing stream generation
    test_output_data = []
    for no_of_notes in range(128):
        test_output_data.append([uniform(-1, 3), uniform(-0.5, 0.5), uniform(0, 0.5)])

    stream = data_to_stream(test_output_data)
    stream_to_file(stream, "../test_output.mid")