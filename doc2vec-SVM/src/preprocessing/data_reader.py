from os import listdir

# ----------------------- DATA LOCATIONS --------------------------------------
DATA_BASEPATH = '../../../data/'
large_dataset_raw_path = DATA_BASEPATH + 'large dataset/raw data/'

# large_dataset_cleaned_path = DATA_BASEPATH + 'large dataset/cleaned data/'
# large_dataset_cleaned_path_all = DATA_BASEPATH + 'large dataset/cleaned data/all_cleaned_data.txt'

# large_dataset_cleaned_path = DATA_BASEPATH + 'large dataset/cleaned data unstemmed/'
# large_dataset_cleaned_path_all = DATA_BASEPATH + 'large dataset/cleaned data unstemmed/all_cleaned_data.txt'

large_dataset_cleaned_path = DATA_BASEPATH + 'large dataset/cleaned data unstemmed pos/'
large_dataset_cleaned_path_all = DATA_BASEPATH + 'large dataset/cleaned data unstemmed pos/all_cleaned_data.txt'

# small_dataset_raw_path = DATA_BASEPATH + 'small dataset/raw data/'
# small_dataset_cleaned_path = DATA_BASEPATH + 'small dataset/cleaned data/'

# small_dataset_raw_path = DATA_BASEPATH + 'small dataset/raw data/'
# small_dataset_cleaned_path = DATA_BASEPATH + 'small dataset/cleaned data unstemmed/'

small_dataset_raw_path = DATA_BASEPATH + 'small dataset/raw data/'
small_dataset_cleaned_path = DATA_BASEPATH + 'small dataset/cleaned data unstemmed pos/'

# ---------------------- doc2vec models ---------------------------------------
trained_doc2vec_models_path = 'doc2vec/doc2vec models/'
# -----------------------------------------------------------------------------

# ---------------------- SVM models ---------------------------------------
svm_doc2vec_crossval_fold_data_path = '../cross_validation/folds/svm_doc2vec/'
svm_crossval_fold_data_path = '../cross_validation/folds/svm/'
# -----------------------------------------------------------------------------


# return contents of file as string
def read_file(filepath):
    return open(filepath, 'r', errors='ignore').read() # reads line?

# write contents to file at filepath
def write_file(filepath, contents):
    file = open(filepath, 'w')
    file.write(contents)
    file.close()