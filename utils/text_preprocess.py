import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

import pandas as pd

def Preprocessing(text):
    """
        1. lower
        2. punctuations are replaced by ' '
        3. tokenization
        4. remove stopwords
        5. stemming
        6. lemmatization
    """
    text = text.lower()  # 将所有的单词转换成小写字母

    for c in string.punctuation:
        text = text.replace(c, " ")  # 将标点符号转换成空格

    wordList = nltk.word_tokenize(text)  # 分词

    filtered = [w for w in wordList if w not in stopwords.words('english')]  # 删除停顿词
    # filtered = [w for w in wordList]
    # stem
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]  # 提取词干
    wl = WordNetLemmatizer()
    filtered = [wl.lemmatize(w) for w in filtered]  # 词形还原
    return " ".join(filtered)

if __name__ == "__main__":
    # mapping = {'Financial': '0',
    #            'Tools': '1',
    #            'Messaging': '2',
    #            'eCommerce': '3',
    #            'Payments': '4',
    #            'Social': '5',
    #            'Enterprise': '6',
    #            'Mapping': '7',
    #            'Science': '8',
    #            'Government': '9'}
    df = pd.read_csv('../data/top_10_api.csv')
    w = open('../data/top_10_api.txt', 'w', encoding='utf-8')

    count = 0.0
    sum_word = 0
    for c, d in zip(df['Primary Category'], df['description']):
        d = Preprocessing(d)
        w.write(c + ',' + d + '\n')
        count += 1
        sum_word += len(d.split(' '))
    w.close()
    print(sum_word/count)
