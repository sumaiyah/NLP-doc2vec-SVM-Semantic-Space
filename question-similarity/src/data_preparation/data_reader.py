import pandas as pd
import numpy as np
from collections import namedtuple, OrderedDict
from typing import List
import json

"""
Contains functionality needed to read in data from csv and explore the data as well as serialize/deserialze cleaned data
"""

# ------------------------------------------- Data locations ------------------------------------------------
raw_data_loc = '../../../data/question-similarity/questions.csv'
clean_data_loc = '../data_preparation/cleaned_data/clean_data_'
clean_data_info_file = '../data_preparation/cleaned_data/info.txt'
# -----------------------------------------------------------------------------------------------------------

# ------------------------------------------- doc2vec locs ------------------------------------------------
doc2vec_models_path = 'doc2vec models/'
doc2vec_models_log_path = 'doc2vec models/info.txt'
# -----------------------------------------------------------------------------------------------------------


QuestionPair = namedtuple('QuestionPair', ['question1_tokens', 'question2_tokens', 'is_duplicate'])


# return dataframe of data being read in
def read_in_data(data_path: str = raw_data_loc, n_items: int = 50000):
    df = pd.read_csv(data_path)
    return df[0:n_items]

def get_av_q_length(n_items: int = 50000):
    path = raw_data_loc
    df = pd.read_csv(path)
    df = df[0:n_items]

    lengths = []
    for i in range(0, len(df)):
        lengths.append(len(df.question1[i].split()))
        lengths.append(len(df.question2[i].split()))

    print('len(lengths): ', len(lengths))
    print('av length: ', (sum(lengths) / len(lengths)))

# print stats about the data to explore contents
def print_data_stats(df: pd.DataFrame):
    # data has id, qid1, qid2, question1, question2, is_duplicate
    # ids start at 0, qids start at 1
    print("Data head\n", df.head())

    print("0: number of non-duplicate pairs, 1: number duplicate pairs\n", df.is_duplicate.value_counts())
    # accuracy won't be as good of a performance metric as F1

    print("\n Number of null items:\n", df.isnull().sum())


# write questionpairlist to file
def write_qpList_to_file(qp_list: List[QuestionPair], filenum: int, clean_params) -> None:
    # clean data location
    qp_file_name = (clean_data_loc + str(filenum) + '.txt')
    qp_file = open(qp_file_name, 'w')

    # log location
    log = open(clean_data_info_file, 'a')

    # write each question pair to file as a json object
    for question_pair in qp_list:
        qp_file.write(json.dumps(question_pair._asdict()))
        qp_file.write('\n')

    # log cleaned file info in txt file
    log.write((qp_file_name) + '\n')
    log.write("parameters: " + str(clean_params))
    log.write('\n\n')


# return list of QuestionPair objects from file
def load_qpList_from_file(filenum) -> List[QuestionPair]:
    question_pair_list = []

    # clean data location
    qp_file_name = (clean_data_loc + str(filenum) + '.txt')

    # list of JSON objects as strings
    qp_JSON_list = open(qp_file_name).readlines()

    # convert each JSON object to a QuestionPair object
    for qp_JSON in qp_JSON_list:
        question_pair_list.append(QuestionPair(**json.loads(qp_JSON)))

    return question_pair_list

