# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:44:05 2019

@author: WT
"""
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import networkx as nx
from collections import OrderedDict
from itertools import combinations
import math
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog-bar")

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
def nCr(n,r):
    f = math.factorial
    return int(f(n)/(f(r)*f(n-r)))

### remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        token = token.lower()
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")", "@",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1

def dummy_fun(doc):
    return doc

def word_word_edges(p_ij):
    word_word = []
    cols = list(p_ij.columns); cols = [str(w) for w in cols]

    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (p_ij.loc[w1,w2] > 0):
            word_word.append((w1,w2,{"weight":p_ij.loc[w1,w2]}))
    return word_word

def generate_text_graph(train_data, max_vocab_len, window=10):
    """ generates graph based on text corpus (columns = (text, label)); window = sliding window size to calculate point-wise mutual information between words """
    logger.info("Preparing data...")
    df = pd.read_csv(train_data, header=0, )#sep='\n')
    df.dropna(inplace=True)

    stopwords = list(set(nltk.corpus.stopwords.words("english")))
        
    ### tokenize & remove funny characters
    df["text_p"] = df["text"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))
    df['length'] = df['text_p'].apply(lambda x: len(x))
    df = df[df['length'] >= (window + 1)] # filter out sents shorter than 4 tokens
    logger.info("Number of documents: %d" % len(df))
    save_as_pickle("df_data.pkl", df)
    
    ### Tfidf
    logger.info("Calculating Tf-idf...")
    vectorizer = TfidfVectorizer(input="content", max_features=max_vocab_len, tokenizer=dummy_fun, preprocessor=dummy_fun)
    vectorizer.fit(df["text_p"])
    df_tfidf = vectorizer.transform(df["text_p"])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf, columns=vocab)
    
    ### Build graph
    doc_nodes = [i for i in range(len(df_tfidf.index))]
    save_as_pickle('doc_nodes.pkl', doc_nodes)
    logger.info("Building graph (No. of document, word nodes: %d, %d)..." %(len(df_tfidf.index), len(vocab)))
    G = nx.Graph()
    logger.info("Adding document nodes to graph...")
    G.add_nodes_from(df_tfidf.index) ## document nodes
    logger.info("Adding word nodes to graph...")
    G.add_nodes_from(vocab) ## word nodes
    ### build edges between document-word pairs
    logger.info("Building document-word edges...")
    document_word = [(doc,w,{"weight":df_tfidf.loc[doc,w]}) for doc in tqdm(df_tfidf.index, total = len(df_tfidf.index))\
                     for w in df_tfidf.columns]
    G.add_edges_from(document_word)
    del df_tfidf, document_word
    
   ### PMI between words
    names = vocab
    n_i  = OrderedDict((name, 0) for name in names)
    word2index = OrderedDict( (name,index) for index,name in enumerate(names) )

    occurrences = np.zeros( (len(names),len(names)) ,dtype=np.int32)
    # Find the co-occurrences:
    no_windows = 0; skipped = 0; not_in_vocab = 0
    logger.info("Calculating co-occurences...")
    for l in tqdm(df["text_p"], total=len(df["text_p"])):
        if len(l) <= window:
            skipped += 1
            continue
        
        for i in range(len(l)-window):
            no_windows += 1
            d = set(l[i:(i+window)])

            for w in d:
                try:
                    n_i[w] += 1
                except: # w not in vocab cos limit
                    continue
                
            for w1,w2 in combinations(d,2):
                if (w1 in names) and (w2 in names):
                    i1 = word2index[w1]
                    i2 = word2index[w2]
    
                    occurrences[i1][i2] += 1
                    occurrences[i2][i1] += 1
                else:
                    not_in_vocab += 1
                    pass
    logger.info("Zero co-occurances calculated for %d documents as they are too short!" % skipped)
    logger.info("No. of skipped co-occurences calculations due to oov: %d" % not_in_vocab)
    
    logger.info("Calculating PMI*...")
    ### convert to PMI
    p_ij = pd.DataFrame(occurrences, index = names,columns=names)/no_windows
    p_i = pd.Series(n_i, index=n_i.keys())/no_windows

    del occurrences
    del n_i
    for col in p_ij.columns:
        p_ij[col] = p_ij[col]/p_i[col]
    for row in p_ij.index:
        p_ij.loc[row,:] = p_ij.loc[row,:]/p_i[row]
    p_ij = p_ij + 1E-9
    for col in p_ij.columns:
        p_ij[col] = p_ij[col].apply(lambda x: math.log(x))
        
    
    logger.info("Building word-word edges...")
    word_word = word_word_edges(p_ij)
    save_as_pickle("word_word_edges.pkl", word_word)
    G.add_edges_from(word_word)
    save_as_pickle("text_graph.pkl", {"graph": G})
    logger.info("Done and saved!")
