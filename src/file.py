import pickle
import music21 as mu


def file_to_stream(path):
    return mu.converter.parse(path).recurse().notes.sorted


def stream_to_file(stream, output_path):
    mf = mu.midi.translate.streamToMidiFile(stream)
    mf.open(output_path, 'wb')
    mf.write()
    mf.close()


def save_file(file_name, data):
    f = open(file_name + ".pkl", "wb")
    pickle.dump(data, f)
    f.close()


def load_file(file_name):
    with open(file_name + ".pkl", 'rb') as f:
        return pickle.load(f)