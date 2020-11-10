from data_preparation import data_reader, data_cleaner
from collections import namedtuple, OrderedDict

from typing import List
import re
"""
Overall data pre-processing
storing read in data into appropriate data structures
"""

# named tuple to hold pairs of questions
QuestionPair = namedtuple('QuestionPair', ['question1_tokens', 'question2_tokens', 'is_duplicate'])

# return List[QuestionPair] regenerating cleaned data if necessary
def preprocess_raw_data(regenerate_data: bool, filenum: int):
    if regenerate_data:
        # read in raw data
        raw_data = data_reader.read_in_data(n_items=50000)

        # list of all question pairs
        question_pair_list = []

        # cleaning parameters same for each question
        clean_params = OrderedDict(remove_stopwords=True, stem=True,
                                   regex_clean=True, lemmatize=False)

        # generate list of question pairs
        for i in range(0, len(raw_data)):
             question_pair_list.append(QuestionPair(
                question1_tokens=data_cleaner.str_to_clean_tokens(raw_data.question1[i], **clean_params),
                question2_tokens=data_cleaner.str_to_clean_tokens(raw_data.question2[i], **clean_params),
                is_duplicate=int(raw_data.is_duplicate[i])))

        # serialise question pair list
        data_reader.write_qpList_to_file(qp_list=question_pair_list, filenum=filenum, clean_params=clean_params)
        return question_pair_list

    else:
        # just load saved clean data
        return data_reader.load_qpList_from_file(filenum=filenum)

# return "valid" question pairs
def remove_invalid_qps(qp_list: List[QuestionPair]) -> List[QuestionPair]:
    valid_qps = []

    n_invalid_qps = 0
    for i in range(len(qp_list)):
        # qs is invalid if either question is empty
        if len(qp_list[i].question1_tokens) == 0 or len(qp_list[i].question2_tokens) == 0:
            n_invalid_qps += 1

        elif (qp_list[i].question1_tokens == qp_list[i].question2_tokens):
            n_invalid_qps += 1

        # questions need to contain a vowel (which should be part of a full word) to be valid
        elif not re.search('[aeiouyAEIOUY]', ' '.join(qp_list[i].question1_tokens)) \
                or not re.search('[aeiouyAEIOUY]', ' '.join(qp_list[i].question2_tokens)):
            n_invalid_qps += 1

        else: # question is valid
            valid_qps.append(qp_list[i])

    print("n_invalid qs: ", n_invalid_qps)
    print("n_valid qs: ", len(valid_qps))

    # should be equal
    print("total original qps: ", len(qp_list), end=' =? ')
    print("total qps: ", (len(valid_qps) + n_invalid_qps))

    return valid_qps

# return qplists where 80% train, 10% validation, 10% test
# TODO CHANGE THIS TO BE ABLE TO DO CROSS VAL FOLD LATER
def get_train_val_test(qp_list: List[QuestionPair]) -> (List[QuestionPair], List[QuestionPair], List[QuestionPair]):
    split_1 = int(0.8 * len(qp_list))
    split_2 = int(0.9 * len(qp_list))

    train = qp_list[:split_1]
    val = qp_list[split_1:split_2]
    test = qp_list[split_2:]

    print('n_train:',len(train), end=' ')
    print('n_val:',len(val), end=' ')
    print('n_test:', len(test))

    return train, val, test

# overall preprocessing return train and test data as question pair lists
def preprocess():
    qp_list = preprocess_raw_data(regenerate_data=False, filenum=1)

    # remove invalid qps
    valid_qps = remove_invalid_qps(qp_list)

    # split into train, val and test
    train_data, val_data, test_data = get_train_val_test(valid_qps)

    return train_data, val_data, test_data


