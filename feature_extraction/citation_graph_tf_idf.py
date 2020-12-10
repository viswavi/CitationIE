import argparse
from collections import defaultdict
from gensim import corpora
from gensim.models import TfidfModel
import json
import jsonlines
from nltk.corpus import stopwords
import numpy as np
import os
import pickle
import scipy
import time

from join_scirex_and_s2orc import (
    S2OrcEntry,
    S2Metadata,
    create_citation_graph_from_seed_nodes,
    get_citation_graph,
    get_scirex_neighbor_texts
)

STOPWORDS_SET = set(stopwords.words('english'))
TERM_DICTIONARY_FILE = "/projects/metis0_ssd/users/vijayv/SciREX/s2orc_caches/fulltexts/term_dictionary.pkl"

def load_full_text_documents(path, num_documents_to_load=None):
    if num_documents_to_load is None:
        # Return lazy reader for full dataset
        return jsonlines.open(path)
    else:
        # Load them into memory and return a list.
        reader = jsonlines.open(path)
        docs = []
        for i, doc in enumerate(reader):
            if i >= num_documents_to_load:
                break
            docs.append(doc)
        return docs


def tokenize_text(full_text_dict):
    all_text = " ".join(section["text"] for section in full_text_dict["body_text"])
    return all_text.lower().split()

def construct_dictionary(document_reader):
    function_start = time.perf_counter()
    dictionary = corpora.Dictionary(tokenize_text(line) for line in document_reader)
    # remove stop words and words that appear only once
    stop_ids = [
        dictionary.token2id[stopword]
        for stopword in STOPWORDS_SET
        if stopword in dictionary.token2id
    ]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    function_end = time.perf_counter()
    print(f"Constructing dictionary took {function_end - function_start} seconds")
    return dictionary

def compute_tf_idf_matrix(document_reader, dictionary):
    function_start = time.perf_counter()
    bows = []
    num_docs = 0
    for i, doc in enumerate(document_reader):
        if i % 10000 == 0:
            print(f"BOW'd up to {i} documents")
        num_docs += 1
        bows.append(dictionary.doc2bow(tokenize_text(doc)))

    tfidf_model = TfidfModel(bows)
    rows = []
    cols = []
    data = []
    doc_tfidf_rows = tfidf_model[bows]
    for i, row in enumerate(doc_tfidf_rows):
        for j, v in row:
            rows.append(i)
            cols.append(j)
            data.append(v)
        
        if i % 10000 == 0:
            print(f"len(data) at doc {i}: {len(data)}")
    
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)

    tf_idf_matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(num_docs, len(dictionary)))
    function_end = time.perf_counter()
    print(f"Computing tf-idf matrix took {function_end - function_start} seconds")
    return tf_idf_matrix

def merge_adjacency_lists(in_adjacencies, out_adjacencies):
    undirected_edges = defaultdict(set)
    for adjacency_list in [in_adjacencies, out_adjacencies]:
        for key, values in adjacency_list.items():
            undirected_edges[key] = undirected_edges[key].union(values)

    undirected_edges_list = {}
    for k, v in undirected_edges.items():
        undirected_edges_list[k] = list(v)
    return undirected_edges_list

def compute_average_term_tfidfs_for_documents(adjacency_list, tf_idf_matrix, doc_idx_mapping):
    term_document_mapping = {}
    for scirex_doc_id, neighbors in adjacency_list.items():
        num_neighbors_with_text = []
        for n in neighbors:
            idx = doc_idx_mapping.get(n, -1)
            #if idx is -1 or <TF IDF matrix at this row has zero entries>:
            #    continue

        term_document_mapping[scirex_doc_id] = {}

    return term_document_mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf-idf-matrix-file', default="/projects/metis0_ssd/users/vijayv/SciREX/s2orc_caches/fulltexts/tfidf.npz",
                        help='.npz file containing full tf-idf matrix')
    parser.add_argument('--out-directory', default="/projects/metis0_ssd/users/vijayv/SciREX/s2orc_caches/fulltexts/tf_idfs/",
                        help='.npz file containing full tf-idf matrix')
    parser.add_argument('--num-documents-to-load', default=None, type=int)
    args = parser.parse_args()

    doc_idx_mapping, scigraph_documents_path = get_scirex_neighbor_texts()

    if os.path.exists(TERM_DICTIONARY_FILE):
        print(f"Loading cached TF-IDF dictionary from {TERM_DICTIONARY_FILE}.")
        tf_idf_dictionary = pickle.load(open(TERM_DICTIONARY_FILE, 'rb'))
    else:
        document_reader = load_full_text_documents(scigraph_documents_path, num_documents_to_load=args.num_documents_to_load)
        tf_idf_dictionary = construct_dictionary(document_reader)
        print(f"Caching TF-IDF dictionary at {TERM_DICTIONARY_FILE}.")
        pickle.dump(tf_idf_dictionary, open(TERM_DICTIONARY_FILE, 'wb'))

    if os.path.exists(args.tf_idf_matrix_file):
        tf_idf_matrix = scipy.sparse.load_npz(args.tf_idf_matrix_file)
        print(f"Loaded TF-IDF matrix from {args.tf_idf_matrix_file}.")
    else:
        document_reader = load_full_text_documents(scigraph_documents_path, num_documents_to_load=args.num_documents_to_load)
        tf_idf_matrix = compute_tf_idf_matrix(document_reader, tf_idf_dictionary)
        scipy.sparse.save_npz(args.tf_idf_matrix_file, tf_idf_matrix)
        print(f"Saved TF-IDF matrix to {args.tf_idf_matrix_file}.")

    (out_edges, in_edges) = get_citation_graph(radius=1)
    undirected_edges = merge_adjacency_lists(out_edges, in_edges)

    adjacency_lists = [out_edges, in_edges, undirected_edges]
    adjacency_names = ["out_edges", "in_edges", "undirected_edges"]

    os.makedirs(args.out_directory, exist_ok=True)
    for graph_connectivity_type, adjacency_list in zip(adjacency_names, adjacency_lists):
        jsonfile = os.path.join(args.out_directory, f"{graph_connectivity_type}.json")
        per_term_tfidfs = compute_average_term_tfidfs_for_documents(adjacency_list, tf_idf_matrix, doc_idx_mapping)
        json.dump(per_term_tfidfs, open(jsonfile, 'w'))
        print(f"Wrote JSON file with citation-graph-averaged TF-IDFs (over {graph_connectivity_type}) to {jsonfile}.")

    from IPython import embed; embed()


if __name__ == "__main__":
    main()