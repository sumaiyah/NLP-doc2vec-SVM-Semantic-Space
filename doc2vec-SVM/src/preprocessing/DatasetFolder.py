from Sentiment import Sentiment

class DatasetFolder():
    def __init__(self, n_folds, doc_filenames_dict):
        self.n_folds = n_folds

        self.doc_filenames = doc_filenames_dict
        self.doc_filenames_folded = {}

        self.fold_dataset()

    def fold_dataset(self):
        for Sent in Sentiment:  # equal distribution for each class
            self.doc_filenames_folded[Sent] = [[] for i in range(0, self.n_folds)]

            # iterate over files of a class and populate each fold
            for i in range(0, len(self.doc_filenames[Sent])):  # add file to the fold
                self.doc_filenames_folded[Sent][i % self.n_folds].append(self.doc_filenames[Sent][i])

    def get_train_and_test_data(self, test_index):
        # test data is the fold at index test_index
        test_data = {Sent: self.doc_filenames_folded[Sent][test_index]
                     for Sent in Sentiment}

        # train data is remaining folds flattened
        train_data = {Sent: [element
                             for fold in self.doc_filenames_folded[Sent]
                             if fold != test_data[Sent]
                             for element in fold]
                      for Sent in Sentiment}

        return train_data, test_data