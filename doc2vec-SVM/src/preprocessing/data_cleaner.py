import re

from nltk.tokenize import word_tokenize  # tokenizer

from nltk.corpus import stopwords       # stop words
import nltk
stop_words = stopwords.words('english')

from nltk.stem.porter import PorterStemmer  # stemmer
ps = PorterStemmer()

html_tag_RE = re.compile(r'<[^>]+>')  # precompiled RegEx to remove HTML tags so only computed once
space_RE = re.compile(r'\s+')   # remove any tokens that are just spaces

punct = [';', ':', '\'', '\"', '*','(', ')',
         '[', ']', '#', '%', '$', '@', '.', ',', '``', '\'\'', '/']  # punctuation to remove

# returns cleaned tokens from raw string
def clean_string_and_tokenize(raw_string):
    # remove html tags
    clean_string = html_tag_RE.sub('', raw_string)

    # remove double spaces and misused periods i.e. replace '  ' with ' '
    clean_string = clean_string.replace('.', ' . ').replace('  ', ' ')

    # remove punctuation
    for punc in punct:
        clean_string = clean_string.replace(punc, ' ')

    # tokenize and clean the tokens
    return clean_tokens(word_tokenize(clean_string))

# returns cleaned tokens
def clean_tokens(raw_tokens):
    # remove stop words
    filtered_tokens = [token for token in raw_tokens if not token in stop_words]

    # remove certain punctuation and normalise case
    clean_tokens = [token.lower() for token in filtered_tokens if not token in punct]

    # remove any tokens that are just whitespace
    clean_tokens = [token for token in clean_tokens if not (re.match(space_RE, token) or token=='')]

    # stem tokens
    # stemmed_tokens = [ps.stem(token) for token in clean_tokens]

    # pos tag tokens
    tagged_tokens = []
    pos_tokens = nltk.pos_tag(clean_tokens)
    for token, pos_tag in pos_tokens:
        tagged_tokens.append(token + '-' + pos_tag)


    # return stemmed_tokens
    # return clean_tokens
    return tagged_tokens

# return cleaned input_text as tokens. Can take string and pre-tokenized data
def clean(input_text):
    if isinstance(input_text, list):    # already tokenized
        cleaned_tokens = clean_tokens(raw_tokens=input_text)
    else:                               # input_text is string
        cleaned_tokens = clean_string_and_tokenize(raw_string=input_text)

    return cleaned_tokens