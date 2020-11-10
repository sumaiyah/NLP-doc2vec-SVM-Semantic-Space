from os import listdir

import data_reader
import data_cleaner
import sentiment_document

# clean and store large dataset
def preprocess_large_dataset():
    # ignoring split in data and using it all as training data for doc2vec

    # process data in train/ -----------------------------------------------------------------
    current_path = data_reader.large_dataset_raw_path + 'train/'

    for sent_class in ['neg', 'pos', 'unsup']:
        for filename in listdir(current_path + sent_class):
            # contents of document
            raw_doc_string = data_reader.read_file(current_path + sent_class + '/' + filename)

            # clean, stem and tokenize
            clean_doc_tokens = data_cleaner.clean(raw_doc_string)

            # save cleaned tokens
            clean_doc_tokens_str = '\n'.join(clean_doc_tokens)  # each token on a new line
            data_reader.write_file((data_reader.large_dataset_cleaned_path + sent_class + '/' + 'tr_' + filename),
                                   clean_doc_tokens_str)
    # ------------------------------------------------------------------------------------------

    print('training data preprocessed')

    # process data in test/ -----------------------------------------------------------------
    current_path = data_reader.large_dataset_raw_path + 'test/'

    for sent_class in ['neg', 'pos']:
        for filename in listdir(current_path + sent_class):
            # contents of document
            raw_doc_string = data_reader.read_file(current_path + sent_class + '/' + filename)

            # clean, stem and tokenize
            clean_doc_tokens = data_cleaner.clean(raw_doc_string)

            # save cleaned tokens
            clean_doc_tokens_str = '\n'.join(clean_doc_tokens)  # each token on a new line
            data_reader.write_file((data_reader.large_dataset_cleaned_path + sent_class + '/' + 'te_' +filename),
                                   clean_doc_tokens_str)    # to keep filename unique

    # ------------------------------------------------------------------------------------------

    print('test data preprocessed')

    # write all to all_cleaned_data.txt --------------------------------------------------------
    open(data_reader.large_dataset_cleaned_path_all, 'w')
    file = open(data_reader.large_dataset_cleaned_path_all, 'a')

    # iterate over all files in large dataset/cleaned data
    for sent_class in ['pos', 'neg', 'unsup']:
        current_path = data_reader.large_dataset_cleaned_path + sent_class + '/'
        for filename in listdir(current_path):
            # each token is on a new line
            doc_tokens = data_reader.read_file(current_path + filename).split('\n')

            # convert sent_class string to numerical value
            sentiment = {'pos': 1.0, 'neg': -1.0, 'unsup': None}[sent_class]

            # write on a new line filename doc_tokens(comma_sep) sentiment
            file.write(filename + ' ' + ','.join(doc_tokens) + ' ' + str(sentiment) + '\n')
    # ------------------------------------------------------------------------------------------

# clean and store small dataset
def preprocess_small_dataset():
    for sent_class in listdir(data_reader.small_dataset_raw_path): # POS or NEG
        current_path = data_reader.small_dataset_raw_path + sent_class + '/'

        for filename in listdir(current_path):
            # contents of current document
            raw_doc_tokens = data_reader.read_file(current_path + filename).split('\n') # each token on new line in original

            # clean and stem
            clean_doc_tokens = data_cleaner.clean(raw_doc_tokens)

            # save cleaned tokens
            clean_doc_tokens_str = '\n'.join(clean_doc_tokens) # each token on a new line
            data_reader.write_file((data_reader.small_dataset_cleaned_path + sent_class + '/' +filename),
                                   clean_doc_tokens_str)


# preprocess_small_dataset()
preprocess_large_dataset()
