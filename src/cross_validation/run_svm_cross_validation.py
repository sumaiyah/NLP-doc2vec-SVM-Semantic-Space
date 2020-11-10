from preprocessing import data_reader
from processing import svm_model
import os

# train and test svm model
def train_and_test(n_folds):
    for fold_index in range (0, n_folds):
        current_fold_path = data_reader.svm_crossval_fold_data_path + 'cv' + str(fold_index) + '/'

        train_data_path = current_fold_path + 'train_data.dat'
        model_path = current_fold_path + 'model.dat'
        test_data_path = current_fold_path + 'test_data.dat'
        test_output_path = current_fold_path + 'test_output.dat'

        # train
        os.system('svm_learn.exe -v 0' + ' ' + train_data_path + ' ' + model_path)

        # test
        os.system('svm_classify.exe -v 0' + ' ' + test_data_path + ' ' + model_path + ' ' + test_output_path)

# return list of all svm results
def get_all_results(n_folds):
    # list of resuts of all n folds
    all_results = []

    # get test results from each fold
    for fold_index in range(0, n_folds):
        # path to svm folds
        current_fold_path = data_reader.svm_crossval_fold_data_path + 'cv' + str(fold_index) + '/'

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
        # print('accuracy on fold ' + str(fold_index) + ': ' + str(accuracy))

    # all_results is a list of lists for each fold
    all_results_flat = [item
                        for fold_results in all_results
                        for item in fold_results]

    av_accuracy = sum(all_results_flat) / len(all_results_flat)
    print('average acc svm: ', av_accuracy)

    return all_results_flat

def run(n_folds):
    # todo change here to change the flags the svm model is tested with
    train_and_test(n_folds)

    # return results from all folds
    return get_all_results(n_folds)

