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
from scipy import sparse
import string
import time

from join_scirex_and_s2orc import (
    S2OrcEntry,
    S2Metadata,
    get_citation_graph,
    get_scirex_neighbor_texts
)

STOPWORDS_SET = set(stopwords.words('english'))
TERM_DICTIONARY_FILE = "s2orc_caches/fulltexts/term_dictionary.pkl"
MEMORY_MAPPING_FILE = "s2orc_caches/fulltexts/bow.mm"

def load_full_text_documents(path, num_documents_to_load=None):
    if num_documents_to_load is None:
        # Return lazy reader for full dataset
        return jsonlines.open(path)
    else:
        # Load them into memory and return a list.
        reader = jsonlines.open(path)
        docs = []
        for i, doc in enumerate(reader):
            if len(docs) >= num_documents_to_load:
                break
            docs.append(doc)
        return docs


def tokenize_text(full_text_dict):
    all_text = " ".join(section["text"] for section in full_text_dict["body_text"])
    words = all_text.lower().split()
    words = [w.strip(string.punctuation) for w in words]
    # Remove punctuation-only words.
    words = [w for w in words if len(w) > 0]
    return words

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

class JsonlinesCorpus:
    '''
    Loads full-text of papers in each row.
    '''
    def __init__(self, fname, dictionary, num_documents_to_load = None):
        self._fname = fname
        self._dictionary = dictionary
        self._num_documents_to_load = num_documents_to_load

    def __iter__(self):
        for i, doc in enumerate(jsonlines.open(self._fname)):
            if self._num_documents_to_load is not None and i >= self._num_documents_to_load:
                break
            # assume there's one document per line, tokens separated by whitespace
            tokenized_text = tokenize_text(doc)
            yield self._dictionary.doc2bow(tokenized_text)


def compute_tf_idf_matrix(documents_path, dictionary, num_documents_to_load=None, memory_mapping_file=MEMORY_MAPPING_FILE):
    function_start = time.perf_counter()
    num_docs = 0
    rows = []
    cols = []
    data = []
    if not os.path.exists(memory_mapping_file):
        mm_start = time.perf_counter()
        print(f"Memory mapping file does not exist, generating now:")
        corpus = JsonlinesCorpus(documents_path, dictionary, num_documents_to_load=num_documents_to_load)
        corpora.MmCorpus.serialize(MEMORY_MAPPING_FILE, corpus, progress_cnt=1000, metadata=True)
        print(f"Generated memory mapping for corpus. Took {time.perf_counter() - mm_start} seconds.")
    else:
        print(f"Memory mapping file already exists at {MEMORY_MAPPING_FILE}, just gonna load that.")

    mm = corpora.MmCorpus(MEMORY_MAPPING_FILE)
    # Build tfidf.
    build_tfidf_start = time.perf_counter()
    tfidf_model = TfidfModel(mm, id2word=dictionary, normalize=True)
    print(f"Building TF-IDF model took {time.perf_counter() - build_tfidf_start} seconds.")

    corpus = JsonlinesCorpus(documents_path, dictionary, num_documents_to_load=num_documents_to_load)
    for i, bow in enumerate(corpus):
        num_docs += 1
        if i % 10000 == 0:
            print(f"BOW'd up to {i} documents")

        doc_tfidf_row = tfidf_model[bow]
        for j, v in doc_tfidf_row:
            rows.append(i)
            cols.append(j)
            data.append(v)
    
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)

    tf_idf_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_docs, len(dictionary)))
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

def compute_average_term_tfidfs_for_documents(adjacency_list, tf_idf_matrix, term_dictionary, doc_idx_mapping):
    function_start = time.perf_counter()
    term_document_mapping = {}
    for scirex_doc_id, neighbors in adjacency_list.items():
        term_document_mapping[scirex_doc_id] = {}
        average_tf_idf = sparse.lil_matrix((1, len(term_dictionary)))

        num_neighbors_with_text = 0.0
        for n in neighbors:
            idx = doc_idx_mapping.get(n, -1)
            if idx is -1 or tf_idf_matrix.data[idx] == []:
                continue
            average_tf_idf += tf_idf_matrix[idx]
            num_neighbors_with_text += 1
        average_tf_idf /= num_neighbors_with_text

        tf_idf_values = {}
        for (i, j), tf_value in average_tf_idf.todok().items():
            token = term_dictionary[j]
            tf_idf_values[token] = tf_value
        term_document_mapping[scirex_doc_id] = tf_idf_values
    function_end = time.perf_counter()
    print(f"Averaging TF-IDFs over neighbors took {function_end - function_start} seconds")
    return term_document_mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear-cache', action='store_true',
                        help="Clear cached files for TF-IDF, which will be regenerated from scratch in this run.")
    parser.add_argument('--tf-idf-matrix-file', default="s2orc_caches/fulltexts/tfidf.npz",
                        help='.npz file to store full tf-idf matrix')
    parser.add_argument('--out-directory', default="s2orc_caches/fulltexts/tf_idfs/")
    parser.add_argument('--num-documents-to-load', default=None, type=int)
    args = parser.parse_args()

    if args.clear_cache:
        if os.path.exists(TERM_DICTIONARY_FILE):
            os.remove(TERM_DICTIONARY_FILE)
            print(f"Removed term dictionary file {TERM_DICTIONARY_FILE} successfully")
        if os.path.exists(MEMORY_MAPPING_FILE):
            os.remove(MEMORY_MAPPING_FILE)
            print(f"Removed corpus memory mapping file {MEMORY_MAPPING_FILE} successfully")
        if os.path.exists(args.tf_idf_matrix_file):
            os.remove(args.tf_idf_matrix_file)
            print(f"Removed TF-IDF matrix file {args.tf_idf_matrix_file} successfully")

    doc_idx_mapping, scigraph_documents_path = get_scirex_neighbor_texts()

    if os.path.exists(TERM_DICTIONARY_FILE):
        print(f"Loading cached TF-IDF dictionary from {TERM_DICTIONARY_FILE}.")
        tf_idf_dictionary = pickle.load(open(TERM_DICTIONARY_FILE, 'rb'))
    else:
        print(f"Going to construct TF-IDF dictionary.")
        document_reader = load_full_text_documents(scigraph_documents_path, num_documents_to_load=args.num_documents_to_load)
        tf_idf_dictionary = construct_dictionary(document_reader)
        print(f"Caching TF-IDF dictionary at {TERM_DICTIONARY_FILE}.")
        pickle.dump(tf_idf_dictionary, open(TERM_DICTIONARY_FILE, 'wb'))

    if os.path.exists(args.tf_idf_matrix_file):
        tf_idf_matrix = sparse.load_npz(args.tf_idf_matrix_file)
        tf_idf_matrix = tf_idf_matrix.tolil()
        print(f"Loaded TF-IDF matrix from {args.tf_idf_matrix_file}.")
    else:
        tf_idf_matrix = compute_tf_idf_matrix(scigraph_documents_path, tf_idf_dictionary, num_documents_to_load=args.num_documents_to_load)
        sparse.save_npz(args.tf_idf_matrix_file, tf_idf_matrix)
        tf_idf_matrix = tf_idf_matrix.tolil()
        print(f"Saved TF-IDF matrix to {args.tf_idf_matrix_file}.")

    (out_edges, in_edges) = get_citation_graph(radius=1, remap_to_scirex_id=True)
    undirected_edges = merge_adjacency_lists(out_edges, in_edges)

    adjacency_lists = [out_edges, in_edges, undirected_edges]
    adjacency_names = ["out_edges", "in_edges", "undirected_edges"]

    os.makedirs(args.out_directory, exist_ok=True)
    for graph_connectivity_type, adjacency_list in zip(adjacency_names, adjacency_lists):
        print(f"Averaging TF-IDFs for {graph_connectivity_type} now")
        jsonfile = os.path.join(args.out_directory, f"{graph_connectivity_type}.json")
        per_term_tfidfs = compute_average_term_tfidfs_for_documents(adjacency_list, tf_idf_matrix, tf_idf_dictionary, doc_idx_mapping)
        json.dump(per_term_tfidfs, open(jsonfile, 'w'))
        print(f"Wrote JSON file with citation-graph-averaged TF-IDFs (over {graph_connectivity_type}) to {jsonfile}.")

if __name__ == "__main__":
    main()