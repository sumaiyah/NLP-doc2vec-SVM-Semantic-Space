from typing import List

from preprocessing import fold_dataset, data_reader
from preprocessing.DatasetFolder import DatasetFolder

path_to_svm_data = data_reader.small_dataset_cleaned_path
path_to_svm_embeddings = data_reader.svm_doc2vec_crossval_fold_data_path

from processing import svm_model

# to store doc2vec embedding for each document
from collections import namedtuple
svmDocEmbedding = namedtuple('svmDocEmbedding', 'filename classification embedding')  # class 1:POS, -1:NEG

# keep a list of all document embeddings to be queries when writing data to each fold
def generate_all_embeddings(doc2vec_model, all_data):
    for key in (all_data.keys()):
        classification = str(key)[-3:]  # either POS or NEG

        # path to cleaned smaller dataset
        data_path = fold_dataset.path_to_data + classification + '/'

        # for each doc, get tokens & infer embedding
        for filename in all_data[key]:
            # tokens for each document
            doc_tokens = data_reader.read_file(data_path + filename).split('\n')

            # infer doc embedding from doc2vec model
            embedding = doc2vec_model.infer_vector(doc_words=doc_tokens, epochs=100, alpha=0.025, min_alpha=0.00005)

            # add correct document classification
            classification_str = ('1' if classification=='POS' else '-1')

            yield(svmDocEmbedding(filename=filename, classification=classification_str, embedding=embedding))

# save embeddings into .dat file in correct place
def save_embeddings(doc_embeddings: List[svmDocEmbedding], filepath: str) -> None:
    file = open(filepath, 'w')  # .dat file

    for doc_embedding in doc_embeddings:
        # generate word vector in the form SVM wants i.e index: value
        doc_token_embedding_str = ''
        count = 1   # SVM light expects tokens with increasing indices
        for token_val in doc_embedding.embedding:
            doc_token_embedding_str += str(count) + ':' + str(token_val) + ' '
            count += 1

        # write to file as classification key:val key:val .....
        file.write(str(doc_embedding.classification) + ' ' + doc_token_embedding_str + '\n')

# if called will regenerate svm data and embeddings and .dat train and test files
def regenerate_svm_data(n_folds, doc2vec_model):
    # entire dataset is now all of the data (90%) we used to train SVM before
    fold_dataset.fold_dataset(n_folds)
    all_data, _ = fold_dataset.get_train_and_validation_data(n_folds - 1)
    SENTIMENTS = list(all_data.keys())  # POS and NEG

    print('generating all word embeddings...', end='')  # just do this once at the start
    all_data_embeddings = list(generate_all_embeddings(doc2vec_model=doc2vec_model, all_data=all_data))
    print('done')

    dataset_folder = DatasetFolder(n_folds, all_data)
    for fold_index in range(0, n_folds):
        # for each fold test_data is fold at fold_index
        train_data, test_data = dataset_folder.get_train_and_test_data(test_index=fold_index)

        # save embeddings to correct.dat file
        train_data_emb = [emb
                          for emb in all_data_embeddings
                          for Sent in SENTIMENTS
                          if emb.filename in train_data[Sent]]
        test_data_emb = [emb
                         for emb in all_data_embeddings
                         for Sent in SENTIMENTS
                         if emb.filename in test_data[Sent]]

        save_embeddings(train_data_emb, (path_to_svm_embeddings + str(fold_index) + '/' + 'train_data.dat'))
        save_embeddings(test_data_emb, (path_to_svm_embeddings + str(fold_index) + '/' + 'test_data.dat'))

# return list of all svm results
def get_all_results(n_folds):
    # list of resuts of all n folds
    all_results = []

    # get test results from each fold
    for fold_index in range(0, n_folds):
        # path to svm folds
        current_fold_path = data_reader.svm_doc2vec_crossval_fold_data_path + str(fold_index) + '/'

        # contents of test_output.dat i.e. svm predictions
        current_test_output = open(current_fold_path + 'test_output.dat', 'r').readlines()
        current_test_output_classifications = [float(item) for item in current_test_output]

        # correct classifications from test_data
        current_test_data = open(current_fold_path + 'test_data.dat', 'r').readlines()
        current_test_data_classifications = [float(row.split(' ')[0]) for row in current_test_data]

        # 1 if predicted correctly else 0
        current_test_results = [1
                                if (current_test_output_classifications[i] * current_test_data_classifications[i]) > 0
                                else 0
                                for i in range(0, len(current_test_output_classifications))]

        # add it to list of all results
        all_results.append(current_test_results)

        accuracy = sum(current_test_results) / len(current_test_results)
        print('accuracy on fold ' + str(fold_index) + ': ' + str(accuracy))

    # all_results is a list of lists for each fold
    all_results_flat = [item
                        for fold_results in all_results
                        for item in fold_results]

    av_accuracy = sum(all_results_flat) / len(all_results_flat)
    print('average acc svm doc2vec: ', av_accuracy)

    return all_results_flat

# run svm cross validation of each of n folds and return list of results from all folds
def run(n_folds, doc2vec_model, regenerate_data_flag):
    if regenerate_data_flag:
        regenerate_svm_data(n_folds, doc2vec_model)

    for fold_index in range(0, n_folds):

        # set parameters for svm model
        svm_model.train_data_path = path_to_svm_embeddings + str(fold_index) + '/train_data.dat'
        svm_model.model_path = path_to_svm_embeddings + str(fold_index) + '/model.dat'
        svm_model.test_data_path = path_to_svm_embeddings + str(fold_index) + '/test_data.dat'
        svm_model.test_output_path = path_to_svm_embeddings + str(fold_index) + '/test_output.dat'

        # svm train and test
        svm_model.run()

    # return results from all folds
    return get_all_results(n_folds)









