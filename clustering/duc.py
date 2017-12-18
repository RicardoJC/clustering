import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

import numpy as np
import pickle

import os
import json

from processing import *

def get_rouge_tokens(original_text):
    '''lemmatized tokens with no stopwords nor punctuation
       Frequency irrelevant
       Capitalized named entities
    '''
    lemmatized = lemmatize(original_text)
    table = str.maketrans(dict.fromkeys(punctuation))
    en_stopwords = set(stopwords.words('english'))
    tokens = [word for word in word_tokenize(lemmatized.translate(table))
              if word not in en_stopwords]
    return set(tokens)

def get_rouge_document_clusters(data_folder_original):
    '''returns documents as rouge tokens, ordered by cluster'''
    cluster_ids = os.listdir(data_folder_original)
    documents = {}

    # Gets the tokens
    for cluster in cluster_ids:
        cluster_folder = data_folder_original + "/" + cluster
        documents[cluster] = {}
        for document in os.listdir(cluster_folder):
            document_path = cluster_folder + "/" + document
            tree = ET.parse(document_path)
            root = tree.getroot()
            original_text = root.find("TEXT").text
            token_set = get_rouge_tokens(original_text)
            documents[cluster][document] = token_set
    return documents

def get_rouge_summary_clusters(data_folder_original):
    '''returns documents as rouge tokens, ordered by cluster'''
    cluster_ids = os.listdir(data_folder_original)
    documents = {}

    # Gets the tokens
    for cluster in cluster_ids:
        cluster_folder = data_folder_original + "/" + cluster
        documents[cluster] = {}
        for document in os.listdir(cluster_folder):
            document_path = cluster_folder + "/" + document
            with open(document_path, "r") as fin:
                original_text = fin.read()
            token_set = get_rouge_tokens(original_text)
            documents[cluster][document] = token_set
    return documents

def convert_to_vectors(documents, vector_space):
    # Calculates the representation
    vectorized_documents = {}
    for cluster in documents:
        vectorized_documents[cluster] = {}
        for document_name in documents[cluster]:
            document = documents[cluster][document_name]
            vectorized_documents[cluster][document_name] = \
                                                    get_vector_representation(
                                                                    document,
                                                                    vector_space)
    return vectorized_documents


'''
Funcion de prueba para agregar la representacion vectorial con tf-idf
Ricardo JC
'''

def convert_to_vectors_for_tf_idf(documents,vector_space):
    vectorized_documents = {}
    for cluster in documents:
        vectorized_documents[cluster] = {}
        for document_name in documents[cluster]:
            document = documents[cluster][document_name]
            vectorized_documents[cluster][document_name]= \
                                                    get_vector_representation_tf_idf(
                                                                        document,
                                                                        documents,
                                                                        vector_space)
    return vectorized_documents


def get_vector_space_from_clusters(documents):
    token_sets = []
    for cluster in documents.keys():
        for token_set in documents[cluster].values():
            token_sets.append(token_set)

    # Calculates the vector space
    space = get_space(token_sets)
    return space

def get_cluster_centroids(vectorized_documents):
    centroids = {}
    for cluster in vectorized_documents:
        centroids[cluster] = get_centroid(tuple(
                                        vectorized_documents[cluster].values()
                                            )
                                )
    return centroids

def jsonify(dictionary):
    if isinstance(dictionary, np.ndarray) or\
       isinstance(dictionary, set):
        return list(dictionary)
    elif isinstance(dictionary, dict):
        for key in dictionary.keys():
            dictionary[key] = jsonify(dictionary[key])
    return dictionary

def dump(dictionary, output_path):
    with open(output_path, "w") as out:
        out.write(json.dumps(jsonify(dictionary)))

def pickle_dump(dictionary, output_path):
    with open(output_path, "wb") as out:
        pickle.dump(dictionary, out)

def pickle_dumps(dictionary, output_path):
    n_bytes = 2**11
    bytes_out = pickle.dumps(dictionary)
    max_bytes = len(bytes_out)-1
    with open(output_path, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
            f_out.flush()
