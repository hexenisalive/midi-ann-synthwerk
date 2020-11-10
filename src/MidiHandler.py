import music21 as mu
import pretty_midi as pm
from os import listdir, remove
from copy import deepcopy
from utilities import print_progress
from file import stream_to_file, save_file, load_file
import threading as th
import queue as qu
import time


class SplitterThread(th.Thread):
    def __init__(self, splitter, queue, th_id):
        th.Thread.__init__(self)
        self.__splitter = splitter
        self.__queue = queue
        self.__id = th_id

    def run(self):
        while not self.__queue.empty():
            file = self.__queue.get()
            self.__splitter.split_midi(file)


class ReassignerThread(th.Thread):
    def __init__(self, reassign_handler, split_path, queue, th_id):
        th.Thread.__init__(self)
        self.__reassign_handler = reassign_handler
        self.__split_path = split_path
        self.__queue = queue
        self.__id = th_id

    def run(self):
        while not self.__queue.empty():
            file = self.__queue.get()
            sequence = mu.converter.parse(self.__split_path + file)
            self.__reassign_handler(sequence, file)

# CLASSES ---------------------------------------


class MidiHandler:
    """
    MidiHandler is a class that prepares MIDI files for further processing.
    The preparation consist of generating music21 sequences for each instrument
    found in input files.
    """
    def __init__(self, threads: int = 2):
        self.__sequences = []
        self.__input_path = "..\\MIDIs\\input\\"
        self.__split_path = "..\\MIDIs\\parts\\"
        self.__regen_path = "..\\MIDIs\\regen\\"
        self.threads = threads
        self.__splitter = MidiSplitter(self.__input_path, self.__split_path)

    def backup(self):
        save_file("mh_seq", self.__sequences)

    def rollback(self):
        self.__sequences = load_file("mh_seq")

    def status(self):
        print("Input files found in: " + self.__input_path + " - (" +
              str(len(listdir(self.__input_path))) + " files found)")
        print("Split files found in: " + self.__split_path + " - (" +
              str(len(listdir(self.__split_path))) + " files found)")
        print("Drum splitting is " + self.__splitter.drum_splitter_status() + ".")
        print("Uniform tempo is " + self.__splitter.uniform_tempo_status() + ".")

    def toggle_drum_splitter(self):
        self.__splitter.extract_drums = not self.__splitter.extract_drums

    def toggle_uniform_tempo(self):
        self.__splitter.uniform_tempo = not self.__splitter.uniform_tempo

    def prepare_data(self):
        """
        Prepares MIDIs, changes them to music21 streams that are then processed and added onto tensorflow tensors.

        Method starts with calling internal MidiSplitter that divides MIDIs into instrumental parts.
        These parts, in order to be readable by music21 toolkit, have to physically exist on hard drive.
        Split MIDIs contain data on time signatures and tempos. MIDI programs (instruments) that are not
        fully supported by music21 module, therefore are stored as comments in music21.instrument.Instrument() classes
        found outside in a dictionary that is then used to reassign these programs when corresponding streams are
        created. These comments become useful in later parts of input data creation process.

        :return:
        """
        # Input files are split into multiple instrument tracks and saved as separate MIDIs.
        input_files = listdir(self.__input_path)

        work_queue = qu.Queue()
        work_threads = []

        t0 = time.time()
        # create work queue
        for file in input_files:
            work_queue.put(file)

        # assign threads to work queue
        for i in range(self.threads):
            thread = SplitterThread(self.__splitter, work_queue, i)
            thread.start()
            work_threads.append(thread)

        # wait till queue finishes
        while not work_queue.empty():
            pass

        # join and clear threads
        for thread in work_threads:
            thread.join()
        work_threads.clear()
        t1 = time.time()
        print("Finished splitting in " + str(t1-t0) + " seconds")

        split_files = listdir(self.__split_path)

        t0 = time.time()
        for file in split_files:
            work_queue.put(file)

        # assign threads to work queue
        for i in range(self.threads):
            thread = ReassignerThread(self.__reassign_program, self.__split_path, work_queue, i)
            thread.start()
            work_threads.append(thread)

        # wait till queue finishes
        while not work_queue.empty():
            pass

        # join and clear threads
        for thread in work_threads:
            thread.join()
        work_threads.clear()
        t1 = time.time()
        print("Finished reassigning in " + str(t1-t0) + " seconds.")

    def regenerate_midis(self, generate_txt: bool = False):
        """
        Regenerate MIDI files from internally stored sequences.

        :param generate_txt: Parameter that controls the generation of .txt files that contain music21.stream data.
        :return:
        """
        self.__clear_files(self.__regen_path)

        pf = "Regenerating MIDI files to " + self.__regen_path + ":"
        end = len(self.__sequences)

        for progress, sequence in enumerate(self.__sequences):
            print_progress(progress, end, prefix=pf, suffix=sequence)
            stream_to_file(sequence, self.__regen_path + "regenerated_" + str(progress) + ".mid")
            if generate_txt:
                file = open(self.__regen_path + "regenerated_" + str(progress) + ".txt", "a")
                for element in sequence[0]:
                    file.write("{" + str(element.offset) + "}" + " - " + str(element) + "\n")
                file.close()
            print_progress(progress + 1, end, prefix=pf, suffix=sequence)

    def get_data(self):
        return self.__sequences

    def clear_data(self):
        """
        Clear previously prepared data.

        :return:
        """
        self.__clear_files(self.__split_path)
        self.__sequences = []
        self.__splitter = MidiSplitter(self.__input_path, self.__split_path)

    def __clear_files(self, clear_path):
        """
        Clear physical files in given directory.

        :param clear_path: Describes a directory that has to be purged.
        :return:
        """
        clear_files = listdir(clear_path)

        pf = "Removing files in " + clear_path + ":"
        end = len(clear_files)

        for progress, file in enumerate(clear_files):
            print_progress(progress, end, prefix=pf, suffix=file)
            remove(clear_path + file)
            print_progress(progress + 1, end, prefix=pf, suffix=file)

    def __reassign_program(self, sequence, file_name: str):
        """
        Reassigns correct music21.instrument for sequence given.

        Sequence iterates through parts (should be one) in order to find music21.instrument
        classes that might have been automatically generated for the sequence. These instruments are removed and
        exchanged for correct music21.instrument.Instrument() that by itself does nothing, but it also contains
        music21.editorial.comments that can be called and detailed MIDI program data can be extracted and later used.

        :param sequence: Music21.stream that contains majority of musical data.
        :param file_name: The name of the file from with the stream was created,
        as well as a dictionary key for its music21.instrument.
        :return:
        """
        for i, part in enumerate(sequence):
            instruments = part.getInstruments()
            for instrument in instruments:
                # Any instruments that could be found in the sequence are removed,
                # as they could potentially be wrongly assigned with multiple occurrences.
                part.remove(instrument)
            # Function tries to reassign the instrument (or MIDI program)
            # that is found in splitter's instrument dictionary.
            try:
                part.insert(0, self.__splitter.instruments_dict[file_name])
            except KeyError:
                print("\nCorresponding key: \"" + file_name + "\" not found. Adding instrument.Instrument().")
                edit = mu.editorial.Editorial()
                edit.true_program = str(0)
                edit.is_drum = str(0)

                insert_instrument = mu.instrument.Instrument()
                insert_instrument.editorial.comments.append(edit)

                part.insert(0, insert_instrument)
            # The sequence might consist of multiple note tracks that are called voices.
            # These tracks are flattened and all music21 elements are put into a single sequence.
            if part.hasVoices():
                sequence[i] = part.flattenUnnecessaryVoices(force=True)


class MidiSplitter:
    """
    MidiSplitter is a class that is capable of dividing complex MIDI files and returning multiple
    files for each instrument track that is found in the input file.
    """
    def __init__(self, input_path: str, split_path: str, extract_drums: bool = True, uniform_tempo: bool = True):
        self.instruments_dict = {}
        self.__input_path = input_path
        self.__split_path = split_path
        self.uniform_tempo = uniform_tempo
        self.extract_drums = extract_drums

    def split_midi(self, file_name: str):
        """
        Split MIDI files to ones corresponding to each instrument found throughout.

        Function creates multiple files for each instrument track and stores them in a separate directory.
        Inclusion of drum tracks is optional. For each track a unique dictionary key is created that has assigned a
        music21.instrument object that can be later referenced in the process of creating learning streams.

        :param file_name: Name of the file found in input path.
        :return:
        """

        # input file is loaded and its initial tempo is extracted
        pretty_mf = pm.PrettyMIDI(self.__input_path + file_name)
        init_tempo = round(pretty_mf.get_tempo_changes()[1][0])

        part_tag = 0
        for instrument in pretty_mf.instruments:
            if self.uniform_tempo:
                # temporary MIDI for a single instrument is created and has copied values from the input MIDI
                temp_pretty = pm.PrettyMIDI(initial_tempo=init_tempo)
                temp_pretty.time_signature_changes = pretty_mf.time_signature_changes
                # a single instrument is appended onto ints instrument list, with it all its notes
                temp_pretty.instruments.append(instrument)
            else:
                # deep copy should preserve all exact tempo changes throughout the song
                temp_pretty = deepcopy(pretty_mf)
                # all existing instruments are cleared and only one from the original file is appended
                temp_pretty.instruments.clear()
                temp_pretty.instruments.append(instrument)

            # assigning comments for music21.instrument
            edit = mu.editorial.Editorial()
            edit.true_program = str(instrument.program)
            edit.is_drum = str(instrument.is_drum)

            insert_instrument = mu.instrument.Instrument()
            insert_instrument.editorial.comments.append(edit)

            # assigning file / dictionary key name
            if not instrument.is_drum:
                out_name = file_name + "_part_" + str(part_tag) + "_" + \
                           pm.program_to_instrument_name(instrument.program) + ".mid"
            else:
                out_name = file_name + "_part_" + str(part_tag) + "_Percussion" + ".mid"

            # placing commented instrument into dictionary for later usage with music21 module
            self.instruments_dict[out_name] = insert_instrument
            part_tag += 1
            temp_pretty.write(self.__split_path + out_name)

    def clear_dict(self):
        self.instruments_dict = {}

    def drum_splitter_status(self):
        if self.extract_drums:
            return "enabled"
        else:
            return "disabled"

    def uniform_tempo_status(self):
        if self.uniform_tempo:
            return "enabled"
        else:
            return "disabled"
