from nltk.tokenize import word_tokenize  # tokenizer
from nltk.corpus import stopwords       # stop words
from nltk.stem.porter import PorterStemmer  # stemmer

import re

from typing import List

"""
Clean actual data items (strings)
"""

stop_words = stopwords.words('english')
ps = PorterStemmer()


# convert string to tokens removing punct, stopwords and stemming/lemmatizing as necessary
def str_to_clean_tokens(raw_str: str, remove_stopwords: bool, regex_clean:bool, stem: bool, lemmatize: bool) -> List[str]:
    # lower case all letters
    raw_str = raw_str.lower()

    # tokenize
    tokens = word_tokenize(raw_str)

    # remove english stopwords
    if remove_stopwords:
        tokens = [token for token in tokens if not token in stop_words]

    clean_str = ' '.join(tokens)

    # Clean up the text
    if regex_clean:
        clean_str = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", clean_str)
        clean_str = re.sub(r"\'s", " 's ", clean_str)
        clean_str = re.sub(r"\'ve", " 've ", clean_str)
        clean_str = re.sub(r"n\'t", " 't ", clean_str)
        clean_str = re.sub(r"\'re", " 're ", clean_str)
        clean_str = re.sub(r"\'d", " 'd ", clean_str)
        clean_str = re.sub(r"\'ll", " 'll ", clean_str)
        clean_str = re.sub(r",", " ", clean_str)
        clean_str = re.sub(r"\.", " ", clean_str)
        clean_str = re.sub(r"!", " ", clean_str)
        clean_str = re.sub(r"\(", " ( ", clean_str)
        clean_str = re.sub(r"\)", " ) ", clean_str)
        clean_str = re.sub(r"\?", " ", clean_str)
        clean_str = re.sub(r"\s{2,}", " ", clean_str)

    tokens = word_tokenize(clean_str)

    # stem words in text
    if stem:
        tokens = [ps.stem(token) for token in tokens]

    # lemmatize tokens in the text
    if lemmatize:
        print("TODO NEED TO ADD LEMMATIZATION FUNCTIONALITY")

    # return tokenized clean string 
    return tokens
