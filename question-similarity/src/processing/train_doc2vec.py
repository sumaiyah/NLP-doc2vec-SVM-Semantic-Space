import multiprocessing
from collections import OrderedDict, namedtuple
from typing import List
from data_preparation import data_reader

from gensim.models.callbacks import CallbackAny2Vec

import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

from gensim.models.doc2vec import Doc2Vec

QuestionPair = namedtuple('QuestionPair', ['question1_tokens', 'question2_tokens', 'is_duplicate'])

"""
train doc2vec model with a portion of the question pairs
"""

# return list of questions as tagged documents ready to be put into doc2vec
def get_train_corpus(qp_list: List[QuestionPair]):
    current_doc_index = 0   # doc tag is zero based index of question
    train_corpus = []
    for qp in qp_list:
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(qp.question1_tokens, [current_doc_index]))
        current_doc_index += 1

        train_corpus.append(gensim.models.doc2vec.TaggedDocument(qp.question2_tokens, [current_doc_index]))
        current_doc_index += 1

    return train_corpus

# build train and save doc2vec model
def train(train_data: List[QuestionPair], model_name: str, parameters):
    # convert train_data to list of tagged documents
    tagged_train_data = get_train_corpus(train_data)
    assert len(tagged_train_data) == 2 * (len(train_data))

    print("n_train_data_qps:", len(train_data), end=' =? ')
    print("n_train_corpus:", len(tagged_train_data))

    # initialise model
    model = Doc2Vec(**parameters, callbacks=[EpochLogger()])

    # train model
    model.build_vocab(tagged_train_data)
    model.train(tagged_train_data, total_examples=model.corpus_count, epochs=model.epochs)

    # save model to disk
    model.save(data_reader.doc2vec_models_path + model_name)

    # log saved model
    log = open(data_reader.doc2vec_models_log_path, 'a')
    log.write(model_name + '\n')
    log.write(str(parameters) + '\n')
    log.write('Accuracy: ' + '\n\n')

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        # print('Epoch #{} start'.format(self.epoch))

        pass

    def on_epoch_end(self, model):
        if self.epoch==0:
            print('Epoch #{}'.format(self.epoch), end=' ')
        else:
            print('#{}'.format(self.epoch), end=' ')
        self.epoch += 1
