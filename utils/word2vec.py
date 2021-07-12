from gensim.models import word2vec
import pandas as pd
import logging

mapping = {'Financial': 0.0,
               'Tools': 1.0,
               'Messaging': 2.0,
               'eCommerce': 3.0,
               'Payments': 4.0,
               'Social': 5.0,
               'Enterprise': 6.0,
               'Mapping': 7.0,
               'Science': 8.0,
               'Government': 9.0}

def read_corpora():
    f = open('../data/top_10_api.txt', 'r', encoding='utf-8')
    data = f.readlines()
    corpora = []
    for item in data:
        item = item.replace('\n', '').split(',')[-1].split(' ')
        corpora.append(item)
    return corpora


if __name__ == "__main__":
    corpora = read_corpora()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(corpora, hs=0, sg=0, negative=10, size=200, window=10, min_count=1, workers=4, iter=25, seed=2019)
    model.save("../model/word2vec.model")

#
# min_word_count = 20   # Minimum word count
# num_workers = 40       # Number of threads to run in parallel
# context = 10          # Context window size
# downsampling = 1e-3   # Downsample setting for frequent words
#
# print "Training Word2Vec model..."
# # Train Word2Vec model.
# model = Word2Vec(sentences, workers=num_workers, hs = 0, sg = 0, negative = 10, iter = 25,\
#             size=num_features, min_count = min_word_count, \
#             window = context, sample = downsampling, seed=1)
