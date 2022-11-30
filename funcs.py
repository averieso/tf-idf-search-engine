#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import xml.sax
from xml.sax import ContentHandler 
import re
import string
from stemming.porter2 import stem
import math
import itertools
import collections
import operator
import numpy as np 
import time
import sys


# # Part 1: Parsing

# In[3]:


class CorpusHandler(xml.sax.ContentHandler):

    data_dict = collections.defaultdict()
    
    def __init__(self):
        self.CurrentData = ""
        self.docid = ""
        self.text = ""

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "DOC":
            self.docid = attributes["id"]
            CorpusHandler.data_dict[self.docid] = {}
            
    def endElement(self, name):
        if name == "DOC":
            CorpusHandler.data_dict[self.docid] = self.text
            self.text = ""
               
    def characters(self, content):
        if content != "\n":
            if self.CurrentData == "HEADLINE" or self.CurrentData == "P" or self.CurrentData == "TEXT":
                self.text += content + " "
                    
    def get_parsed_data(self):
        return CorpusHandler.data_dict       


# # Part 2:  Pre-processing

# In[4]:


def remove_ws(token):
    return token.strip()

def remove_puncs(token):
    return token.translate(str.maketrans('', '', string.punctuation))


# In[5]:


def preprocess(text):

    if type(text) != list:
        # lower case
        text = text.lower()
        
        # split into tokens at whitespace 
        text = text.split(" ")
    
        text = list(map(remove_ws, text))
        
        text = list(map(remove_puncs, text))
        
        text = filter(None, text)
        
        # stemming
        text = list(map(stem, text))
    
    return list(text)

# In[6]:


def token_to_numeric(preprocessed_dict):
    total_n_words = list(itertools.chain.from_iterable(preprocessed_dict.values()))
    
    # get every unique token in the corpus
    vocab = sorted(set(total_n_words))
    if "" in vocab:
        return print("this corpus contains empty strings.")
    print("there are ", len(vocab), "unique words in this corpus.")
    
    # assign an index to every unique token in the corpus
    word2num_dict = {}
    for i, token in enumerate(vocab):
        word2num_dict[token] = i
    
    return word2num_dict


# In[7]:


def parse(path_to_xml_filename, run_preprocess = True):
    print('Parsing xml file...')
    parser = xml.sax.make_parser()

    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    Handler = CorpusHandler()

    parser.setContentHandler(Handler)
    parser.parse(path_to_xml_filename)

    #get parsed data in the form of a dict, with doc id as key
    parsed_dict = Handler.get_parsed_data()
    
    print('there are', len(parsed_dict), 'documents.')
    
    #preprocessing
    if run_preprocess == True:
        print('Preprocessing data...')
        preprocessed_dict = {doc: preprocess(parsed_dict[doc]) for doc in parsed_dict}
    
        return preprocessed_dict
    
    else:
        return parsed_dict


# # Part 3: tf-idf

# In[8]:


def idf(preprocessed_dict, ind2word = True, save_idf = False):
    print('creating idf index...')
    n_of_docs = len(preprocessed_dict)
    word2ind_dict = token_to_numeric(preprocessed_dict)
    
    #empty np array to store idf values for each unique word
    idf_array = n_of_doc_token = np.zeros(len(word2ind_dict))
    idf_dict = {}
    
    for doc in preprocessed_dict:
        for token in set(preprocessed_dict[doc]):
            word2ind = word2ind_dict[token]
            n_of_doc_token[word2ind] += 1
    
    idf_array = np.log(n_of_docs / n_of_doc_token)
    
    if save_idf == True:
        filename = input("Please input file name for this idf index: ")
        with open("%s.idf" %filename,"w") as f:
            if ind2word == True:    
                for word, index in word2ind_dict.items():
                    idf_dict[word] = idf_array[index]
                    line = word + "\t" + str(idf_dict[word]) + "\n"
                    f.write(line)
        return idf_dict
    else:
        if ind2word == True:    
            for word, index in word2ind_dict.items():
                idf_dict[word] = idf_array[index]
            return idf_dict
        else:
            return idf_array


# In[9]:


def tf(token, document):
    n_token_in_doc = document.count(token)
    maxOccurrences = collections.Counter(document).most_common(1)[0][1]
    tf_value = n_token_in_doc / maxOccurrences
    return tf_value


# In[10]:


def tf_corpus(preprocessed_dict, ind2word = True, save_tf = False):
    print('creating tf index...')
    word2ind_dict = token_to_numeric(preprocessed_dict)
    
    tf_dict = {}
    tf_array = np.zeros((len(word2ind_dict), len(preprocessed_dict)))
    for i, docname in enumerate(preprocessed_dict.keys()):
        document = preprocessed_dict[docname]
        term_freq = collections.Counter(document)
        maxOccurrence = term_freq.most_common(1)[0][1]
        if ind2word == True:
            temp_array = np.array(list(term_freq.values())) / maxOccurrence
            tf_dict[docname] = dict(sorted(zip(term_freq.keys(), temp_array)))
        else:
            for token in term_freq.keys():
                word2ind = word2ind_dict[token]
                tf_array[word2ind, i] = term_freq[token] / maxOccurrence

    if ind2word == True:
        if save_tf == True:
            filename = input("Please input file name for this tf index: ")
            with open("%s.tf" %filename,"w") as f:  
                for docname in tf_dict: 
                    for token in tf_dict[docname]:
                        line = docname + "\t" + token + "\t" + str(tf_dict[docname][token]) + "\n"
                        f.write(line)
        return tf_dict
    else:
        return tf_array


# In[11]:


def tfidf(token, doc_name, tf_dict, idf_dict):
    return tf_dict[doc_name][token] * idf_dict[token]


# # Part 4: Creating the Search Engine

# In[13]:


def similarity(query, document, tf_dict, idf_dict):
    dotp = 0
    sum_d = 0
    sum_q = 0
    for q in query:
        tf_q = tf(q, query)
        try:  
            idf_q = idf_dict[q]
        except KeyError:
            idf_q = 0
        try:
            tfidf_d = tfidf(q, document, tf_dict, idf_dict)
        except KeyError:
            tfidf_d = 0
        tfidf_q = tf_q * idf_q
        sum_q += tfidf_q ** 2
        dotp += tfidf_q * tfidf_d
    for token, tf_value in tf_dict[document].items():
        sum_d += (tf_value * idf_dict[token]) ** 2
    
    norm_d = math.sqrt(sum_d)
    norm_q = math.sqrt(sum_q)
    
    res = dotp / (norm_q * norm_d)
  
    return res









