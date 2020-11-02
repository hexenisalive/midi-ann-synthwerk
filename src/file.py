import pickle as pkl
import music21 as mu


def stream_to_file(stream, output_path):
    mf = mu.midi.translate.streamToMidiFile(stream)
    mf.open(output_path, 'wb')
    mf.write()
    mf.close()


def save_file(file_name, data):
    f = open("../pkl_files/" + file_name + ".pkl", "wb")
    pkl.dump(data, f)
    f.close()


def load_file(file_name):
    with open("../pkl_files/" + file_name + ".pkl", 'rb') as f:
        return pkl.load(f)
