from nltk.tokenize import sent_tokenize
from pycorenlp import StanfordCoreNLP
from nltk.cluster import util

import error_metrics as err

import numpy as np

nlp= StanfordCoreNLP("http://localhost:9000")

def lemmatize(original_text):
    sentences = sent_tokenize(original_text)
    processed_sentences = []
    for sentence in sentences:
        output = nlp.annotate(sentence,
                              properties={
                                          'annotators': 'lemma',
                                          'outputFormat': 'json'
                              }
        )
        lemmas = []
        for token in output['sentences'][0]['tokens']:
            lemmas.append(token['lemma'])
        processed_sentences.append(" ".join(lemmas))
    return " ".join(processed_sentences)

def get_space(token_sets):
    space = set()
    for token_set in token_sets:
        space = space.union(token_set)
    return sorted(list(space))

def get_centroid(vector_cluster):
    return np.mean(vector_cluster, axis=0)

# get seed centroids
def get_initial_centroids(vect_docs, ideal_centroids):
    init_centroids = np.array([])
    for cluster in vect_docs:
        candidate = None
        min_distance = None
        for document in vect_docs[cluster]:
            if min_distance == None:
                min_distance = util.cosine_distance(
                                                ideal_centroids[cluster],
                                                vect_docs[cluster][document])
                candidate = vect_docs[cluster][document]
            else:
                candidate_distance = util.cosine_distance(
                                                ideal_centroids[cluster],
                                                vect_docs[cluster][document])
                if candidate_distance < min_distance:
                    min_distance = candidate_distance
                    candidate = vect_docs[cluster][document]
        if len(init_centroids) == 0:         
            init_centroids = candidate
        else:
            #import ipdb
            #ipdb.set_trace() 
            #init_centroids.append(init_centroids)
            init_centroids = np.vstack([init_centroids, candidate])
            #init_centroids.append(candidate) 
    return init_centroids
