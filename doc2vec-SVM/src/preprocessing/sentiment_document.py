import collections
from os import listdir

import data_reader

# DataType for holding a single document
# words:        text of document as a list of words
# id:           unique index of document in entire dataset i.e. filename
# sentiment:    1 (positive), -1 (negative), None (unlabeled document)
SentimentDocument = collections.namedtuple('SentimentDocument', 'words tags sentiment')

def get_sentiment_documents():
    # iterate over all files in large dataset/cleaned data

    # data about each file is on a new line
    all_docs = data_reader.read_file(data_reader.large_dataset_cleaned_path_all).split('\n')

    for doc in all_docs:
        # each line has filename file_tokens(comma sep) sentiment
        doc_information = doc.split(' ')

        # SORT OUT
        # assert len(doc_information)==3, 'all_cleaned_data.txt invalid format'
        if (len(doc_information) == 3):
            filename = doc_information[0]
            doc_tokens = doc_information[1].split(',')
            sentiment = {'1.0': 1.0, '-1.0': -1.0, 'None': None}[doc_information[2]]

            yield SentimentDocument(doc_tokens, [filename], sentiment)
