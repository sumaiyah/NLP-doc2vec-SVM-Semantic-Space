from preprocessing import fold_dataset, data_reader
from preprocessing.DatasetFolder import DatasetFolder
import preprocessing.fold_dataset
from preprocessing.Sentiment import Sentiment
from typing import List

from preprocessing.data_reader import small_dataset_raw_path

path_to_svm_data = data_reader.small_dataset_cleaned_path
path_to_svm_embeddings = data_reader.svm_doc2vec_crossval_fold_data_path

from processing import svm_model

# to store doc2vec embedding for each document
from collections import namedtuple
svmDocEmbedding = namedtuple('svmDocEmbedding', 'filename classification embedding')  # class 1:POS, -1:NEG

def get_all_correct_classifications(n_folds):
    # list of resuts of all n folds
    all_fold_correct_classifications = []

    # get test results from each fold
    for fold_index in range(0, n_folds):
        # path to svm folds
        current_fold_path = data_reader.svm_doc2vec_crossval_fold_data_path + str(fold_index) + '/'

        # correct classifications from test_data
        current_test_data = open(current_fold_path + 'test_data.dat', 'r').readlines()
        current_fold_correct_classifications = [float(row.split(' ')[0]) for row in current_test_data]

        # add it to list of all results
        all_fold_correct_classifications.append(current_fold_correct_classifications)

    # all_results is a list of lists for each fold
    all_fold_correct_classifications_flat = [item
                        for fold_results in all_fold_correct_classifications
                        for item in fold_results]

    return all_fold_correct_classifications_flat

def get_all_predicted_classifications(n_folds):
    # list of resuts of all n folds
    all_fold_pred_classifications = []

    # get test results from each fold
    for fold_index in range(0, n_folds):
        # path to svm folds
        current_fold_path = data_reader.svm_doc2vec_crossval_fold_data_path + str(fold_index) + '/'

        # correct classifications from test_data
        current_test_output = open(current_fold_path + 'test_output.dat', 'r').readlines()
        current_fold_pred_classifications = [float(item) for item in current_test_output]

        # add it to list of all results
        all_fold_pred_classifications.append(current_fold_pred_classifications)

    # all_results is a list of lists for each fold
    all_fold_pred_classifications_flat = [item
                                             for fold_results in all_fold_pred_classifications
                                             for item in fold_results]

    return all_fold_pred_classifications_flat

# return indicies of documents predicted incorrectly as false positives and false negatives
def get_fp_and_fn():
    fp, fn = 0, 0
    fp_indicies, fn_indicies = [], []

    # fp predicted positive but actually negative
    # fn prediced negative but actialu positive

    for i in range(0, len(pred_classifications)):
        if (pred_classifications[i] > 0) and (correct_classifications[i] < 0):
            fp += 1
            fp_indicies.append(i)
        elif (pred_classifications[i] < 0) and (correct_classifications[i] > 0):
            fn += 1
            fn_indicies.append(i)

    print('n false pos: ',fp)
    print('n false neg: ',fn)

    return fp_indicies, fn_indicies

# return indicies of top n misclassified docs
def get_top_n_misclassified(misclassified_doc_indicies, n):
    # the further it is from the margin the more confident svm was that it was correct even though it was wrong
    diffs = [abs(pred_classifications[i]) for i in misclassified_doc_indicies]
    diffs_sorted = diffs.copy()
    diffs_sorted.sort()

    print("Max differences: ", diffs_sorted[-n:])
    max_diff_indices = [misclassified_doc_indicies[diffs.index(diffs_sorted[i])] for i in range(len(diffs) - n, len(diffs))] # last n in sorted list

    return max_diff_indices

# return indicies of bottom n misclassified docs i.e. closest to the margin
def get_botton_n_misclassified(misclassified_doc_indicies, n):
    # the further it is from the margin the more confident svm was that it was correct even though it was wrong
    diffs = [abs(pred_classifications[i]) for i in misclassified_doc_indicies]
    diffs_sorted = diffs.copy()
    diffs_sorted.sort()


    print("Min differences: ", diffs_sorted[:n])
    min_diff_indices = [misclassified_doc_indicies[diffs.index(diffs_sorted[i])] for i in range(0, n)]  # first n in sorted list

    return min_diff_indices



# ------------------------------------------------------------------------------------------------------------------
# return the file name and sent in the folded dataset given an overall index
def get_filename_and_sent_by_overall_index(index):
    Sent = Sentiment((index // 90) % 2)

    if Sent is Sentiment.NEG:
        index = index - (90 * (index//90//2))
    else:
        index = index - (90 * ((index //90//2) + 1))


    fold_dataset.fold_dataset(10)
    data, _ = fold_dataset.get_train_and_validation_data(0)

    if Sent is Sentiment.POS:
        sent = list(data.keys())[1]
    else:
        sent = list(data.keys())[0]

    # file name of file
    return (data[sent][index]), Sent

# return contents of a file as a string
def get_file_contents(filename, sent):
    if sent is Sentiment.POS:
        path = small_dataset_raw_path + 'POS/'
    else:
        path = small_dataset_raw_path + 'NEG/'

    return str(open(path + filename).readlines())

# def write contents of all files with given indicies to txt file for analyis
def write_to_fp_fn_file(fp: bool, top:bool, file_indicies: List[int]):
    # open correct file to write to
    if fp:
        path = 'fp/'
        if top:
            f = open(path + 'top.txt', 'w')
        else:
            f = open(path + 'bottom.txt', 'w')

    else:
        path = 'fn/'
        if top:
            f = open(path + 'top.txt', 'w')
        else:
            f = open(path + 'bottom.txt', 'w')

    for index in file_indicies:
        filename, sent = get_filename_and_sent_by_overall_index(index)
        f.write(filename + str(sent) + '\n')
        f.write(get_file_contents(filename, sent) + '\n \n')

pred_classifications = get_all_predicted_classifications(10)
correct_classifications = get_all_correct_classifications(10)

# ------------------------------------------------------------------------------------------------------------------
fp_indicies, fn_indicies = get_fp_and_fn()

print()
print('false pos: ')
fp_top = (get_top_n_misclassified(misclassified_doc_indicies=fp_indicies, n=5))

print('false neg: ')
fn_top = (get_top_n_misclassified(misclassified_doc_indicies=fn_indicies, n=5))

print('false pos: ')
fp_bottom = (get_botton_n_misclassified(misclassified_doc_indicies=fp_indicies, n=5))

print('false neg: ')
fn_bottom = (get_botton_n_misclassified(misclassified_doc_indicies=fn_indicies, n=5))
print()
# ------------------------------------------------------------------------------------------------------------------

write_to_fp_fn_file(fp=True, top=True, file_indicies=fp_top)
write_to_fp_fn_file(fp=False, top=True, file_indicies=fn_top)
write_to_fp_fn_file(fp=True, top=False, file_indicies=fp_bottom)
write_to_fp_fn_file(fp=False, top=False, file_indicies=fn_bottom)
