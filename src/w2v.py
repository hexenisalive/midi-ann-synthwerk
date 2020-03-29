from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from file import load_file, save_file


def build_word2vec_model(plot: bool = False, annotate: bool = False):
    print("building model...")

    master_dict = load_file("dictionary")
    sequence_dict = load_file("sequences")
    sequence_data = sequence_dict["sequences"]
    # based on sequence data generate word embeddings
    # where sentences are sequences of dictionary keys ('ChordG5G6', 'NoteD7', etc.)
    w2v_model = Word2Vec(sequence_data, min_count=1)

    # projecting vectors onto 2d space
    X = w2v_model[w2v_model.wv.vocab]
    coords = PCA(n_components=2).fit_transform(X)

    # adding coordinates to corresponding keys in global dictionary
    words = list(w2v_model.wv.vocab)
    for coord, word in zip(coords, words):
        master_dict[word]["coords"] = list(coord)

    print("saving updated dictionary...")
    save_file("dictionary", master_dict)

    vocab = {"coords": list(coords), "words": words}
    save_file("w2v_vocab", vocab)

    # optional code snippet that plots the word embedding in 2d space
    if bool(plot):
        plt.scatter(coords[:, 0], coords[:, 1])
        if bool(annotate):
            for i, word in enumerate(words):
                plt.annotate(word, xy=(coords[i, 0], coords[i, 1]))
        plt.show()