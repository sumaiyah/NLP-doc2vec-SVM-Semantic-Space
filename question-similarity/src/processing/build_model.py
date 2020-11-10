from data_preparation import data_preprocessing
from processing import train_doc2vec
from typing import List
from collections import namedtuple
import random
from random import randint
from analysis.similarity import calculate_cosine_distance
from data_preparation.data_reader import doc2vec_models_path
from gensim.models.doc2vec import Doc2Vec

import multiprocessing
"""
split data into train and test set
build doc2vec model
build triplets for testing
test model on triplets
"""
import numpy as np

# 2 questions where is_duplicate is true if the questions are asking the same thing
QuestionPair = namedtuple('QuestionPair', ['question1_tokens', 'question2_tokens', 'is_duplicate'])

# 2 questions are similar third question is a random question
QuestionTriplet = namedtuple('QuestionTriplet', ['sim_question1_tokens', 'sim_question2_tokens', 'odd_question_tokens'])

# 2 questions are similar third question is a random question - store VECTORS
VectorTriplet = namedtuple('VectorTriplet', ['sim_q1_vector', 'sim_q2_vector', 'odd_q_vector'])

# return list of qp triplets given qp_list used for test/val data
def generate_question_triplet_list(qp_list: List[QuestionPair]) -> List[QuestionTriplet]:
    q_triplet_list = []

    # List of question pairs where the 2 questions are asking the same thing
    duplicate_qps = [qp for qp in qp_list if qp.is_duplicate]

    # List of question pairs where the 2 questions are not asking the same thing
    non_duplicate_qps = [qp for qp in qp_list if not qp.is_duplicate]

    # both should add up to length of original qp_list
    assert len(duplicate_qps)+len(non_duplicate_qps)==len(qp_list)

    for duplicate_qp in duplicate_qps:
        # the odd question to add to the triplet randomly chosen
        odd_qp = random.choice(non_duplicate_qps)

        # randomly choose whether to add first q or second q from q pair
        odd_question_tokens = odd_qp.question1_tokens if randint(0,1) else odd_qp.question2_tokens

        q_triplet_list.append(QuestionTriplet(
                            sim_question1_tokens=duplicate_qp.question1_tokens,
                            sim_question2_tokens=duplicate_qp.question2_tokens,
                            odd_question_tokens=odd_question_tokens))

    print("\n n_duplicate_qps: ", len(duplicate_qps), end= ' =? ')
    print("n_q_triplet_list: ", len(q_triplet_list))

    return q_triplet_list

# return results as list [0,1,1,..,] where 0 indicates if classified incorrectly and 1 if correct
def test_doc2vec_model(model_name, test_data: List[QuestionTriplet]):
    # load model from disk
    model = Doc2Vec.load(doc2vec_models_path + model_name)

    # number of triplets where vectors of similar qs are closer together
    n_correct = 0
    results = []

    for question_triplet in test_data:
        vec_triplet = VectorTriplet(
            sim_q1_vector=model.infer_vector(question_triplet.sim_question1_tokens),
            sim_q2_vector=model.infer_vector(question_triplet.sim_question2_tokens),
            odd_q_vector=model.infer_vector(question_triplet.odd_question_tokens))

        # cd -> cosine distances
        cosine_distances = [None]*3 # cd[0]=cos(0-1), cd[1]=cos(1-2), cd[2]=cos(0-3)
        cosine_distances[0] = calculate_cosine_distance(vec_triplet.sim_q1_vector, vec_triplet.sim_q2_vector)
        cosine_distances[1] = calculate_cosine_distance(vec_triplet.sim_q2_vector, vec_triplet.odd_q_vector)
        cosine_distances[2] = calculate_cosine_distance(vec_triplet.sim_q1_vector, vec_triplet.odd_q_vector)

        # correct if cd[0] is smallest as 0,1 in triplet are the similar questions
        if (cosine_distances[0]<cosine_distances[1]) and (cosine_distances[0]<cosine_distances[2]):
            n_correct += 1
            results.append(1)
        else:
            results.append(0)

    print("n_correct: ", n_correct)
    print("total n_triplets: ", len(test_data))
    print("accuracy: ", n_correct/len(test_data))

    return results


def tune_doc2vec_hyperparams(train_data_qps: List[QuestionPair], val_data_qps: List[QuestionPair]):
    # model name
    model_name = 'model_4.model'

    # doc2vec model parameters
    parameters = dict(dm=0, vector_size=100, epochs=30, min_count=1, workers=multiprocessing.cpu_count(), negative=5,
                      hs=0, sample=0)

    # train doc2vec model until best accuracy on val_data
    train_doc2vec.train(train_data_qps, model_name=model_name, parameters=parameters)

    # take best doc2vec model and test accuracy on test_data
    results = test_doc2vec_model(model_name=model_name, test_data=generate_question_triplet_list(val_data_qps))

# print nearest neighbours in doc2vec space
def print_nearest_neighbours(model_name):
    model = Doc2Vec.load(doc2vec_models_path + model_name)
    ms = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=10)
    for x in ms:
        print(x[0], x[1])


        # trained_model.most_similar(positive=['woman', 'king'], negative=['man'])

def count_words_method(test_data: List[QuestionTriplet]):
    acc = 0
    for triplet in test_data:
        q1_words = set(triplet.sim_question1_tokens)
        q2_words = set(triplet.sim_question2_tokens)
        odd_q_words = set(triplet.odd_question_tokens)

        interect12 = q1_words.intersection(q2_words)
        intersect1o = q1_words.intersection(odd_q_words)
        intersect2o = q2_words.intersection(odd_q_words)
        #
        # print('q12: ', len(interect12))
        # print('q1o: ', len(intersect1o))
        # print('q1o: ', len(intersect2o))
        # print()

        if (len(interect12) > len(intersect1o)) and (len(interect12) > len(intersect2o)):
            acc += 1

    print(acc/len(test_data))

def run():
    # split data
    train_data_qps, val_data_qps, test_data_qps = data_preprocessing.preprocess()

    triplets = generate_question_triplet_list(test_data_qps)
    count_words_method(triplets)

    # tune doc2vec hyper params testing the accuracy on the validation set
    # tune_doc2vec_hyperparams(train_data_qps, val_data_qps)

    # test best model on test data
    # test_doc2vec_model(model_name='model_109.model', test_data=generate_question_triplet_list(test_data_qps))

    # print(test_data_qps[70].question1_tokens)
    # print_nearest_neighbours(model_name='model_4.model')

    # try count word method
    # count_words_method(test_data=generate_question_triplet_list(test_data_qps))


def draw_scatter(model_name, tsne_coords, colours):
    import matplotlib.patheffects as PathEffects
    from gensim.models.doc2vec import Doc2Vec
    import numpy as np
    from sklearn.manifold import TSNE

    import seaborn as sns
    from matplotlib import pyplot as plt
    # choose colour palette with seaborn
    num_classes = len(np.unique(colours))
    # assert num_classes == 20
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1], lw=0, s=40, c=palette[colours.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    plt.title('T-SNE of Queston Pairs')

    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(tsne_coords[colours == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=10)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    model_fig_name = model_name.split('.')[0] + '_fig.png'
    plt.savefig(('figs/' + model_fig_name), dpi=120)


def draw_tsne(model_name, doc_vectors, labels):
    from sklearn.manifold import TSNE
    tsne_coords = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca', perplexity=4).fit_transform(doc_vectors)
    #     tsne_coords = TSNE(n_components=2).fit_transform(doc_vectors)

    colours = np.asarray([int(labels[i]) for i in range(0, len(labels))])

    draw_scatter(model_name, tsne_coords, colours)

def draw():
    m = Doc2Vec.load('doc2vec models/model_3.model')
    vecs = []
    labels = []

    qp_list, _, _ = data_preprocessing.preprocess()

    x = 0
    for qp in qp_list:
        if qp.is_duplicate:
            if x > 15: break
            vecs.append(m.infer_vector(qp.question1_tokens))
            vecs.append(m.infer_vector(qp.question2_tokens))

            x += 1


    #
    # vecs.append(m.infer_vector("Why are so many Quora users posting questions that are readily answered on Google?".split()))
    # vecs.append(m.infer_vector("Why do people ask Quora questions which can be answered easily by Google?".split()))
    #
    # vecs.append(m.infer_vector("How do I prepare for civil service?".split()))
    # vecs.append(m.infer_vector("How do we prepare for UPSC?".split()))
    #
    # vecs.append(m.infer_vector("I was suddenly logged off Gmail. I can't remember my Gmail password and just realized the recovery email is no longer alive. What can I do?".split()))
    # vecs.append(m.infer_vector("I can't remember my Gmail password or my recovery email. How can I recover my e-mail?".split()))
    #
    # vecs.append(m.infer_vector("What is Java programming? How To Learn Java Programming Language ?".split()))
    # vecs.append(m.infer_vector("How do I learn a computer language like java?".split()))
    #
    # vecs.append(m.infer_vector("Who is the richest gambler of all time and how can I reach his level as a gambler?".split()))
    # vecs.append(m.infer_vector("Who is the richest gambler of all time and how can I reach his level?".split()))
    #
    # vecs.append(m.infer_vector("".split()))

    print("len: ", len(vecs))
    for i in range(0,len(vecs)//2):
        labels.append(i)
        labels.append(i)

    labels = np.array(labels)

    draw_tsne(model_name='test', doc_vectors=vecs, labels=labels)

draw()
