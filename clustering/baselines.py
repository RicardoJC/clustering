import duc

data_folder_original = "../res/original"
data_folder_summ10 = "../res/summ10"
data_folder_summ100 = "../res/summ100"

output_folder_original = "../outputs/original"
output_folder_summ10 = "../outputs/summ10"
output_folder_summ100 = "../outputs/summ100"

# Text dumps
out_documents_file = "/rouge_docs.txt"
out_rouge_space_file = "/rouge_space.txt"
out_vectorized_documents_file = "/rouge_vect_docs.txt"
out_centroids_file = "/rouge_centroids.txt"

#Pickled dumps
p_documents_file = "/rouge_docs.p"
p_rouge_space_file = "/rouge_space.p"
p_vectorized_documents_file = "/rouge_vect_docs.p"
p_centroids_file = "/rouge_centroids.p"

# Rouge centroids
documents = duc.get_rouge_document_clusters(data_folder_original)
duc.dump(documents, output_folder_original + out_documents_file)
duc.pickle_dump(documents, output_folder_original + p_documents_file)

rouge_space = duc.get_vector_space_from_clusters(documents)
duc.dump(rouge_space, output_folder_original + out_rouge_space_file)
duc.pickle_dump(rouge_space, output_folder_original +  p_rouge_space_file)

vectorized_documents = duc.convert_to_vectors(documents, rouge_space)
duc.dump(vectorized_documents, output_folder_original + out_vectorized_documents_file)
duc.pickle_dump(vectorized_documents, output_folder_original + p_vectorized_documents_file)

centroids = duc.get_cluster_centroids(vectorized_documents)
duc.dump(centroids, output_folder_original + out_centroids_file)
duc.pickle_dump(centroids, output_folder_original + p_centroids_file)
