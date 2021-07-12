# The source code for WSR.

Python Version: 3.5.6

library: 

nltk 3.3.0

numpy 1.14.2

gensim 3.4.0

sklearn 0.20.0

### You need to create a directory for storing word2vec.model.

Usage(generating wsr):
1. mkdir model      //creating a directory for storing word2vec.model by youself.

2. python word2vec.py       //obtaining word embedding for word in vocabulary.   (This isn't word2vec baseline method, just a step in the proposed method)

3. python wsr.py       //obtaining web service representation based on step 2. And evaluating the result of the proposed method.

This repo only contain wsr project code! The code of baseline methods can be found as following:

tfidf: https://radimrehurek.com/gensim/models/tfidfmodel.html

lda: https://radimrehurek.com/gensim/models/ldamodel.html

word2vec: https://radimrehurek.com/gensim/models/word2vec.html

doc2vec: https://radimrehurek.com/gensim/models/doc2vec.html

twe-1: https://github.com/thunlp/topical_word_embeddings

igba: https://gitee.com/qiaoxiang/igba

scdv: https://github.com/dheeraj7596/SCDV

