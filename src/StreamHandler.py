import tensorflow as tf
from utilities import print_progress
from file import save_file, load_file


class StreamHandler:
    def __init__(self, sequences: list):
        self.__sequences = sequences
        self.__input = []
        self.__target = []

    def load_sequences(self, sequences):
        self.__sequences = sequences

    def backup(self):
        save_file("sh_in", self.__input)
        save_file("sh_tar", self.__target)

    def rollback(self):
        self.__input = load_file("sh_in")
        self.__target = load_file("sh_tar")

    def prepare_data(self, subsequence_length: int = 16, subsequence_width: int = 6):
        """
        Final training data preparations. Final lists are created that can be converted into tensorflow.Tensor classes.

        Function iterates through loaded sequences, extracts necessary parameters
        and creates measures from these sequences. These measures are then transformed into
        two-dimensional lists of size [subsequence_length + 1, subsequence_width].
        Subsequence_length describes how many notes can be stored in one measure
        (+1 is for tag_line that sits in the beginning of every subsequence and gives more
        information concerning the measure structure).

        :param subsequence_length:
        :param subsequence_width:
        :return:
        """
        # data is reset
        self.__input = []
        self.__target = []

        if subsequence_width >= 6:

            pf = "Splitting parts to subsequences: "
            end = len(self.__sequences)

            for progress, sequence in enumerate(self.__sequences):
                print_progress(progress, end, prefix=pf, suffix=sequence)
                # divide each sequence into measures that are defined by time signature
                measures = sequence[0].makeMeasures()

                # save initial tempo (dynamically changing tempos are not supported)
                metronome = float(sequence[0].metronomeMarkBoundaries()[0][2].number)
                # saving initial time signature
                time_signature = sequence[0].timeSignature

                # saving tag line values instrument values
                instrument = sequence[0].getInstrument()
                is_drum = float(instrument.editorial.comments[0].is_drum == "True")
                program = float(instrument.editorial.comments[0].true_program)

                # subsequences are unique to each midi track,
                # these same subsequences will be divided into training and target data
                # (subsequent subsequences)
                subsequences = []

                for measure in measures:
                    # creating subsequence for each measure
                    subsequence = []

                    # deleting empty measure and skipping to the next one
                    if measure.highestOffset == 0.0:
                        measures.remove(measure)
                        continue

                    # detecting changes in time signatures
                    if measure.timeSignature is not None:
                        time_signature = measure.timeSignature

                    # tag_line is a list composed of "instruction values"
                    # that are found at the very beginning of every input vector
                    tag_line = [float(measure.offset), program, is_drum, metronome,
                                float(time_signature.numerator), float(time_signature.denominator)]
                    tag_line += ([0.0] * (subsequence_width - 6))
                    subsequence.append(tag_line)

                    # adding elements to subsequence
                    for element in measure.notes.sorted:
                        line = [float(element.offset), float(element.quarterLength)]
                        # adding cord pitches to width
                        for pitch in element.pitches:
                            line.append(float(pitch.midi))

                        # zero-padding width
                        diff = subsequence_width - len(line)
                        for i in range(abs(diff)):
                            if diff > 0:
                                line.append(0.0)
                            else:
                                line.pop()
                                print("Subsequence width (" + str(subsequence_width) + ") may result in data losses.")
                        subsequence.append(line)

                    # zero-padding length
                    diff = subsequence_length + 1 - len(subsequence)
                    for i in range(abs(diff)):
                        if diff > 0:
                            subsequence.append([0.0] * subsequence_width)
                        else:
                            subsequence.pop()
                            print("Subsequence length (" + str(subsequence_length) + ") may result in data losses.")

                    subsequences.append(subsequence)

                # assigning input and target data to subsequent subsequences
                expected_offset = 0.0
                past_subsequence = None
                for subsequence in subsequences:
                    tag_line = subsequence[0]
                    current_offset = tag_line[0]
                    if current_offset == expected_offset and past_subsequence is not None:
                        self.__input.append(past_subsequence)
                        self.__target.append(subsequence)
                    past_subsequence = subsequence
                    expected_offset = current_offset + (tag_line[4]/tag_line[5]*4)

                print_progress(progress+1, end, prefix=pf, suffix=sequence)
        else:
            print("Subsequence width (" + str(subsequence_width) + ") cannot be smaller than 6")

    def get_data(self):
        # data is converted into tensorflow tensors
        return tf.convert_to_tensor(self.__input, dtype=tf.float32), \
               tf.convert_to_tensor(self.__target, dtype=tf.float32)

    def clear_data(self):
        self.__input = []
        self.__target = []