from nltk.tokenize import sent_tokenize
from pycorenlp import StanfordCoreNLP
from nltk.cluster import util

import error_metrics as err

import numpy as np
import math

nlp= StanfordCoreNLP("http://localhost:9000")



'''
tf-idf por partes 
'''

def tf(word,document):
    return sum(1 for w in document if w == word)/len(document)

'''
dtf
Se puede optimizar porque el valor de los documentos se puede obtener en duc.py
sin hacerlo nuevamente en processing.py
'''
def dtf(word,documents):
    total = 0
    for cluster in documents:
        for document_name in documents[cluster]:
            document = documents[cluster][document_name]
            if word in document:
                total = total + 1
    return total

def idf(word,documents):
    d = 0
    for cluster in documents:
        d = d + len(documents[cluster])
    return math.log( d / (1+(dtf(word,documents))))                   # Mas uno para evitar el denominador con 0

def tfidf(word,document,documents):
    ''' Pruebas
    print ('Word: %s' % word)
    print (document)
    print ('tf: %f' % tf(word,document))
    print ('dft: %d' % dtf(word,documents))
    print ('idf: %f' % idf(word,documents))
    print ('tfidf %f' % (tf(word,document) * idf(word,documents)))
    '''
    return tf(word,document) * idf(word,documents)


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

# if a token is in space, it will set a 1 in the space's index
def get_vector_representation(token_set, space):
    vector = np.zeros(len(space))
    for token in token_set:
        if token in space:
            vector[space.index(token)] = 1.0
    return vector

'''
 Representacion vectorial con TF-IDF
 Se hace una representacion vectorial por documento
 Otra opcion podria ser tener una sola por cada palabra
 Hay palabras que no estan en el vector space como: guaranty
'''
def get_vector_representation_tf_idf(document,documents,vector_space):
    vector = np.zeros(len(vector_space))
    for word in document:
        if word in vector_space:
            vector[vector_space.index(word)] = tfidf(word,document,documents)
    return vector

'''
documents is a cluster of documents

def get_document_vectors(documents, rouge_space):
    # Calculates the representation
    vectorized_documents = {}
    for cluster in documents:
        vectorized_documents[cluster] = {}
        for document_name in documents[cluster]:
            document = documents[cluster][document_name]
            vectorized_documents[cluster][document_name] = \
                                                    get_vector_representation(
                                                                    document,
                                                                    rouge_space)
    return vectorized_documents

'''


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

                #min_distance = util.euclidean_distance(len(ideal_centroids[cluster]),len(vect_docs[cluster][document]))
                candidate = vect_docs[cluster][document]
            else:

                candidate_distance = util.cosine_distance(
                                                ideal_centroids[cluster],
                                                vect_docs[cluster][document])
                #candidate_distance = util.euclidean_distance(len(ideal_centroids[cluster]),len(vect_docs[cluster][document]))


                if candidate_distance < min_distance:
                    min_distance = candidate_distance
                    candidate = vect_docs[cluster][document]
        if len(init_centroids) == 0:
            init_centroids = candidate
        else:
            init_centroids = np.vstack([init_centroids, candidate])
    return init_centroids
