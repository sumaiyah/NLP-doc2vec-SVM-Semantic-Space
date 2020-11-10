from Sentiment import Sentiment
import data_reader
from os import listdir

path_to_data = data_reader.small_dataset_cleaned_path

doc_filenames = {  # load dataset into dictionary
            Sentiment.POS: listdir(path_to_data + 'POS/'),
            Sentiment.NEG: listdir(path_to_data + 'NEG/')
        }

doc_filenames_folded = {}

# returns dictionary of filenames split into n_folds using stratified round robin technique
def fold_dataset(n_folds):
    for Sent in Sentiment:  # equal distribution for each class
        doc_filenames_folded[Sent] = [[] for i in range(0, n_folds)]

        # iterate over files of a class and populate each fold
        for i in range(0, len(doc_filenames[Sent])):  # add file to the fold
            doc_filenames_folded[Sent][i % n_folds].append(doc_filenames[Sent][i])

# flatten folded dataset
def flatten_dataset(dataset):
    return {Sent: [element
                         for fold in dataset[Sent]
                         for element in fold]
                  for Sent in Sentiment}

# return training and validatiton data given folded data
def get_train_and_validation_data(val_index):
    # validation data is the fold at index val_index
    val_data = {Sent: doc_filenames_folded[Sent][val_index]
                 for Sent in Sentiment}

    # train data is remaining folds flattened
    train_data = {Sent: [element
                         for fold in doc_filenames_folded[Sent]
                         if fold != val_data[Sent]
                         for element in fold]
                  for Sent in Sentiment}

    return train_data, val_data
