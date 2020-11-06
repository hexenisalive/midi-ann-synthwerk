# MIDI files copyright holder
# Name: Bernd Krueger Source: http://www.piano-midi.de

# Colin Raffel and Daniel P. W. Ellis.
# Intuitive Analysis, Creation and Manipulation of MIDI Data with pretty_midi.
# In 15th International Conference on Music Information Retrieval Late Breaking and Demo Papers, 2014.

from MidiHandler import MidiHandler
from StreamHandler import StreamHandler

if __name__ == "__main__":
    handler = MidiHandler()
    handler.clear_data()
    handler.toggle_uniform_tempo()
    handler.prepare_data()
    s = handler.get_data()

    handler2 = StreamHandler(s)
    handler2.prepare_data()
    x, y = handler2.get_data()

