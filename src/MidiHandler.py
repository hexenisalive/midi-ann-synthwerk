import music21 as mu
import pretty_midi as pm
from os import listdir, remove
from utilities import print_progress
from file import stream_to_file, save_file, load_file


class MidiHandler:
    """
    MidiHandler is a class that prepares MIDI files for further processing.
    The preparation consist of generating music21 sequences for each instrument
    found in input files.
    """
    def __init__(self):
        self.__sequences = []
        self.__input_path = "..\\MIDIs\\input\\"
        self.__split_path = "..\\MIDIs\\parts\\"
        self.__regen_path = "..\\MIDIs\\regen\\"
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

    def toggle_drum_splitter(self):
        self.__splitter.extract_drums = not self.__splitter.extract_drums

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

        pf = "Splitting MIDI files:"
        end = len(input_files)

        for progress, file in enumerate(input_files):
            print_progress(progress, end, prefix=pf, suffix=file)
            self.__splitter.split_midi(file)
            print_progress(progress + 1, end, prefix=pf, suffix=file)

        # Split files are turned into music21 sequences.
        split_files = listdir(self.__split_path)

        pf = "Assigning instruments for parts:"
        end = len(split_files)

        for progress, file in enumerate(split_files):
            print_progress(progress, end, prefix=pf, suffix=file)
            sequence = mu.converter.parse(self.__split_path + file, quantizePost=True)
            # Processed sequence is now reassigned its instrument that have potentially lost during conversion.
            self.__reassign_program(sequence, file)
            self.__sequences.append(sequence)
            print_progress(progress + 1, end, prefix=pf, suffix=file)

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
        print("Removing sequences.")
        self.__sequences = []
        print("Resetting splitter.")
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
    def __init__(self, input_path: str, split_path: str, extract_drums: bool = True):
        self.instruments_dict = {}
        self.__input_path = input_path
        self.__split_path = split_path
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
            # temporary MIDI for a single instrument is created and has copied values from the input MIDI
            temp_pretty = pm.PrettyMIDI(initial_tempo=init_tempo)
            temp_pretty.time_signature_changes = pretty_mf.time_signature_changes
            # a single instrument is appended onto ints instrument list, with it all its notes
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
