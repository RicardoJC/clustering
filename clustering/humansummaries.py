import pickle
import duc
from nltk.cluster import util

import numpy as np

#with open("../outputs/original/rouge_docs.p", "rb") as fin:
#    docs = pickle.load(fin)
#with open("../outputs/original/rouge_vect_docs.p", "rb") as fin:
#    vect_docs = pickle.load(fin)
with open("../outputs/original/rouge_space.p", "rb") as fin:
    space = pickle.load(fin)
#with open("../outputs/original/rouge_centroids.p", "rb") as fin:
#    ideal_centroids = pickle.load(fin)

data_folder_summ10 = "../res/summ10"
data_folder_summ100 = "../res/summ100"

output_folder_summ10 = "../outputs/summ10"
output_folder_summ100 = "../outputs/summ100"

out_documents_file = "/rouge_docs.txt"
out_vectorized_documents_file = "/rouge_vect_docs.txt"

p_documents_file = "/rouge_docs.p"
p_vectorized_documents_file = "/rouge_vect_docs.p"

# Rouge summ10
documents = duc.get_rouge_summary_clusters(data_folder_summ10)
duc.dump(documents, output_folder_summ10 + out_documents_file)
duc.pickle_dump(documents, output_folder_summ10 + p_documents_file)

vectorized_documents = duc.convert_to_vectors(documents, space)
duc.dump(vectorized_documents, output_folder_summ10 + out_vectorized_documents_file)
duc.pickle_dumps(vectorized_documents, output_folder_summ10 + p_vectorized_documents_file)

# Rouge summ100
#documents = duc.get_rouge_summary_clusters(data_folder_summ100)
#duc.dump(documents, output_folder_summ100 + out_documents_file)
#duc.pickle_dump(documents, output_folder_summ100 + p_documents_file)
#
#vectorized_documents = duc.convert_to_vectors(documents, space)
#duc.dump(vectorized_documents, output_folder_summ100 + out_vectorized_documents_file)
#duc.pickle_dump(vectorized_documents, output_folder_summ100 + p_vectorized_documents_file)
