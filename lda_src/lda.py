import numpy as np
import pandas as pd
import time

from gensim.models import Phrases
from gensim.corpora import Dictionary

import cppimport

from lda_src.numba_gibbs import griffiths_steyvers_numba
from lda_src.cython_gibbs import griffiths_steyvers_cython

from stop_words import get_stop_words
import string
from collections import Counter
import os, sys

# this is needed for making sure we are importing the C++ code from the correct path.
# cppimport.imp() looks everywhere in sys.path, so just add the installed package in that path at function call.
if not any('lda_src' in sp for sp in sys.path):
    sys.path.append('lda_src')

for sp in sys.path:
    if os.path.isdir(sp + '/lda_src'):
        sys.path.append(sp + '/lda_src')


class LDA(object):
    def __init__(self, init_topics: int, init_alpha: float, init_beta: float) -> None:
        self.T = init_topics
        self.alpha = init_alpha
        self.beta = init_beta
        self.lda_gibbs = cppimport.imp("gibbs")
        self.vocabulary = None
        self.theta = None
        self.phi = None

    def clean_docs(self, docs):
        '''

        This function cleans the strings provided for the LDA.
        We recommend a more tailored data cleaning according to the user's needs.

        This function also includes biagrams in the documents.

        :param docs: list of strings. Documents to be used in the LDA
        :return: corpus: Gensim corpus format
        :return: vocabulary: a dict with the word_index: word
        :return: dictionary: Gensim dictionary format
        '''

        # (STR)
        strange_symbols = "–’…‘“”—"
        # (RE) Replace strange symbols, with an apostrophe (will be removed later, after stopwords)
        docs = list(map(lambda x: x.translate(str.maketrans(strange_symbols, "'" * len(strange_symbols))),
                        docs))  # strange apostrophes. replace with regular apostrophe

        # (STR) in order to remove stop_words line I've, I'm, You're... keep apostrophe
        new_punctuation = string.punctuation.replace("'", "") + string.digits  # this is not needed if I do lemmatization before

        # (RM) Remove - punctuation, digits, lower_case
        docs = list(map(lambda x: x.lower().replace('-', ' ').translate(str.maketrans(new_punctuation, ' ' * len(new_punctuation))), docs))

        # (RM) Remove - whitspaces
        docs = [' '.join(s.split()) for s in docs]

        # (RM) Remove - more than 3 words in a document
        docs = [s for s in docs if len(s.split()) > 3]

        ############################################################################################################################################
        # Remove stop words

        # (DF) List of docs -> Dataframe: for now I find it faster at replacing words
        docs_df = pd.DataFrame(docs)

        #####################
        en_stop = get_stop_words('en')

        stopwords = set(list(en_stop))

        # (RE) Replace - Stopwords. Must sort the stop_words otherwise using '\b' replaces I before I'm.
        # Can also try with \s+ but problem if two stop words in a row
        regx_en_stop = r'(\b{}\b)'.format(r'\b|\b'.join(sorted(stopwords, key=lambda x: -len(x))))
        docs_df[0] = docs_df[0].replace(regx_en_stop, '', regex=True).replace(r' +', ' ', regex=True).str.strip()

        ############################################################################################################################################

        # (DATA) DataFrame copy of the main data
        docs = docs_df[0].tolist()

        # (RE) Remove - basically only the "'" - apostrophe - is remaining
        docs = list(map(lambda x: x.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))), docs))  # apostrophe,
        # would miss something like 'don't' when replacing with whitespace

        # (RE) Remove - words with less than N letter
        docs = [' '.join([w for w in d.split() if (len(w) > 2)]) for d in docs]

        # (DEL) Documents with less than 3 words in a document
        docs = [s.split() for s in docs if len(s.split()) > 3]

        bigram = Phrases(docs, min_count=20)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    docs[idx].append(token)

        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=20, no_above=0.85)

        vocabulary = dict(dictionary)

        corpus = [dictionary.doc2bow(doc) for doc in docs]

        return corpus, vocabulary, dictionary

    def parse_docs(self, docs: list) -> list:
        '''
        this function formats the data in a compressed format, similar to the scipy COO format
        :param docs: list of strings, documents to be processed
        :return: documents corpus
        :return: vocabulary
        :return: gensim dictionary
        :return: the COO sparse corpus_np
        '''

        corpus, vocabulary, dictionary = self.clean_docs(docs)

        docs, words, words_counts = [], [], []

        for doc_idx, doc in enumerate(corpus):
            docs.extend([doc_idx] * len(doc))
            words.extend([k for k, v in doc])
            words_counts.extend([v for k, v in doc])

        corpus_np = np.r_[np.array(docs)[None, :], np.array(words)[None, :], np.array(words_counts)[None, :]]

        return corpus, vocabulary, dictionary, corpus_np

    def summarize_docs(self, corpus):
        '''
        this function creates the data that is used by the implementation of the algorithm.
        it 'expands' the compressed format into the input for the LDA.
        :param corpus:
        :return: docs_id: an array with documents IDs for as many words as in a document
        :return: all_words: words IDs for all the words of a document (including repetitions)
        :return: docs_nwords: number of word for each document (including repetitions)
        '''

        docs = corpus[0]
        words = corpus[1]
        word_counts = corpus[2, :]

        all_words = np.repeat(words, word_counts)
        docs_id = np.repeat(docs, word_counts)

        df = pd.DataFrame(np.c_[docs, word_counts])
        docs_nwords = np.array(df.groupby(0)[1].sum())

        return docs_id, all_words, docs_nwords

    def init_assignments(self, all_words, docs_id):
        '''
        This function initialized the assignements of the topics for each word (ta), the word-topics matrix (wt) and
        the topic-document matrix (dt).
        :param all_words: array of words IDs for all the words of a document (including repetitions)
        :param docs_id:
        :return: ta
        :return: wt
        :return: dt
        '''

        # print('start init')
        # start_init = time.time()

        ta = np.random.choice(self.T, size=len(all_words))

        wt = pd.DataFrame(np.c_[ta, all_words], columns=['ta', 'w_idx'])
        wt = wt.groupby(['ta', 'w_idx']).size().reset_index()
        wt = np.array(pd.pivot_table(wt, values=0, index=['ta'], columns=['w_idx']).fillna(0).astype(int))

        dt = pd.DataFrame(np.c_[docs_id, ta], columns=['d_idx', 'ta'])
        dt = dt.groupby(['d_idx', 'ta']).size().reset_index()
        dt = np.array(pd.pivot_table(dt, values=0, index=['d_idx'], columns=['ta']).fillna(0).astype(int))

        # print('end dt')
        # print((time.time() - start_init) / 60)

        return ta, wt, dt

    def build_topics_df(self, vocab):
        '''
        function builds dataframe for the found topics including words and associated weights
        :param vocab: vocabulary
        :return: dataframe topic-word-weight
        '''

        df = pd.DataFrame(self.phi)
        df.columns = list(vocab.values())
        df['topic'] = list(range(self.T))

        self.df = pd.melt(df, id_vars='topic', var_name='word', value_name='weight')

    def topic_results(self, topic, words=10):
        '''
        method to be called by the user to display the topics.

        :param topic: topic number
        :param words: number of words to be included in the output
        :return: dataframe with ordered words in order of importance for the selected topic
        '''
        return self.df[self.df['topic'] == topic].sort_values(by='weight', ascending=False).head(words)

    def numba_gibbs(self, iterations, docs_nwords, all_words, ta, wt, dt):
        '''
        Numba implementation of the collapsed gibbs sampler as in Griffiths and Steyvers (2004)

        :param iterations: number of iterations chosen by the user. Iterations for the collapsed gibbs sampler
        :param docs_nwords: array of number of word for each document (including repetitions)
        :param all_words: array of words IDs for all the words of a document (including repetitions)
        :param ta: numpy array with the initialized topic assignments for each word
        :param wt: initialized word-topic matrix
        :param dt: initialized topic-document matrix
        :return: updated wt, dt
        '''
        return griffiths_steyvers_numba(iterations, self.alpha, self.beta, self.D, self.W, self.T, docs_nwords, all_words, ta, wt, dt)

    def cpp_gibbs(self, iterations, docs_nwords, all_words, ta, wt, dt):
        '''
        C++ implementation of the collapsed gibbs sampler as in Griffiths and Steyvers (2004)

        :param iterations: number of iterations chosen by the user. Iterations for the collapsed gibbs sampler
        :param docs_nwords: array of number of word for each document (including repetitions)
        :param all_words: array of words IDs for all the words of a document (including repetitions)
        :param ta: array with the initialized topic assignments for each word
        :param wt: initialized word-topic matrix
        :param dt: initialized topic-document matrix
        :return: updated wt, dt
        '''
        return self.lda_gibbs.griffiths_steyvers(iterations, self.alpha, self.beta, self.D, self.W, self.T, docs_nwords, all_words, ta, wt, dt)

    def cython_gibbs(self, iterations, docs_nwords, all_words, ta, wt, dt):
        '''
        Cython implementation of the collapsed gibbs sampler as in Griffiths and Steyvers (2004)

        :param iterations: number of iterations chosen by the user. Iterations for the collapsed gibbs sampler
        :param docs_nwords: array of number of word for each document (including repetitions)
        :param all_words: array of words IDs for all the words of a document (including repetitions)
        :param ta: array with the initialized topic assignments for each word
        :param wt: initialized word-topic matrix
        :param dt: initialized topic-document matrix
        :return: updated wt, dt
        '''
        return griffiths_steyvers_cython(iterations, self.alpha, self.beta, self.D, self.W, self.T, docs_nwords, all_words, ta, wt, dt)

    def check_assertions(self, method):
        '''
        Checks the input method corresponds with the implemented methods
        '''
        assert method in ["cpp", "numba", "cython"], "method argument can only be one of 'cpp', 'numba', and 'cython'!"

    def initialize(self, docs):
        '''
        When called by the used initializes the Word-Topic Matrix and the Topic-Document Matrix
        it contains many of the attributes defined above and that will be useful to the user
        in for a better document exploration as well as interfacing with other packages without the need
        to process the data again.
        '''

        print("Processing documents... ", end="")

        corpus, vocabulary, dictionary, corpus_np = self.parse_docs(docs)

        docs_id, all_words, docs_nwords = self.summarize_docs(corpus_np)

        ta, wt, dt = self.init_assignments(all_words, docs_id)

        self.ta = ta
        self.wt = wt
        self.dt = dt

        self.corpus = corpus
        self.vocabulary = vocabulary
        self.dictionary = dictionary

        self.docs_id = docs_id
        self.docs_nwords = docs_nwords
        self.all_words = all_words

        self.term_frequency = [v for k, v in Counter(list(all_words)).items()]
        self.vocab = [v for k, v in vocabulary.items()]
        self.doc_length = list(docs_nwords)

        print("DONE!\n")

    def fit(self, iterations, tune_hyperparameters=True, method="cpp"):
        '''
        Fits three implementations of the LDA algorithm with Collapsed Gibbs Sampling. It call a separate file according to the
        selected method: C++ implementation, the Numba implementation and the cython implementation.
        '''

        self.check_assertions(method)

        self.W = len(self.vocabulary)  # number of words in vocabulary
        self.D = len(self.docs_nwords)

        if tune_hyperparameters == True:
            self.alpha = 50 / self.T
            self.beta = 200 / self.W

        print("Starting Collapsed Gibbs Sampler")

        t1 = time.time()
        if method == "cpp":
            wt, dt = self.cpp_gibbs(iterations, self.docs_nwords, self.all_words, self.ta, self.wt, self.dt)
        elif method == "numba":
            wt, dt = self.numba_gibbs(iterations, self.docs_nwords, self.all_words, self.ta, self.wt, self.dt)
        else:
            wt, dt = self.cython_gibbs(iterations, self.docs_nwords, self.all_words, self.ta, self.wt, self.dt)

        print(f"Completed in {time.time() - t1:.2f}s")

        # topic probabilities per document
        self.theta = (dt + self.alpha) / (dt + self.alpha).sum(axis=1).reshape(-1, 1)
        # topic probabilities per word
        self.phi = (wt + self.beta) / (wt + self.beta).sum(axis=1).reshape(-1, 1)
        self.build_topics_df(self.vocabulary)


