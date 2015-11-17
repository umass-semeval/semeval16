from gensim.models import word2vec


def load_embeddings(filename):
    model = word2vec.Word2Vec.load_word2vec_format(filename, binary=True)
    return model
