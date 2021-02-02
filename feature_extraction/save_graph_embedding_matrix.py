'''
python save_graph_embedding_matrix.py \
--graph_embeddings_pickle SciREX/graph_embeddings/graph.pkl \
--node_to_id_mapping node_id_mapping.json \
--out_embedding_file SciREX/graph_embeddings/embeddings.npy \
--out_doc_to_index_mapping_file SciREX/graph_embeddings/scirex_docids.json
'''

import argparse
import json
import pickle
import numpy as np
from join_scirex_and_s2orc import S2Metadata, S2OrcEntry, get_scirex_to_s2orc_mappings, get_scirex_docids
from tqdm import tqdm


def determine_document_index_mapping(all_docids):
    doc_idx_mapping = dict(zip(all_docids, range(len(all_docids))))
    return doc_idx_mapping

def write_scirex_index_mapping_to_file(scirex_doc_idx_mapping, mapping_file):
    json.dump(scirex_doc_idx_mapping, open(mapping_file, 'w'), indent=4)
    print(f"Wrote SciREX-to-embedding-index mapping to {mapping_file}")

def unpack_embeddings(path):
    graph = pickle.load(open(path, 'rb'))
    embeddings_matrix = graph['solver']['vertex_embeddings']
    name2id = graph['graph']['name2id']
    return embeddings_matrix, name2id


def construct_embedding_matrix(full_graph_embedding_matrix, doc_id_to_graph_idx_mapping, doc_id_to_new_idx):
    '''
    full_graph_embedding_matrix: Numpy array of graph embeddings
    doc_id_to_graph_idx_mapping: Dictionary mapping S2orc (?) document ID to row index in full_graph_embedding_matrix
    doc_id_to_new_idx:           Dictionary mapping each S2orc document ID to a new graph embedding matrix containing
                                 only the requested document ids
    '''
    _, embedding_dim = full_graph_embedding_matrix.shape
    embedding_size = len(doc_id_to_new_idx)
    embedding_matrix = np.zeros((embedding_size, embedding_dim))

    # Compute the average embedding among SciREX documents in graph embedding space, to use
    # for documents without a graph embedding.
    average_embedding = np.zeros(embedding_dim)
    nonzero_embedding_count = 0
    for full_graph_embedding_idx in doc_id_to_graph_idx_mapping.values():
        if full_graph_embedding_idx is -1:
            continue
        average_embedding += full_graph_embedding_matrix[full_graph_embedding_idx]
        nonzero_embedding_count += 1
    average_embedding = average_embedding / float(nonzero_embedding_count)

    for doc_id, full_graph_embedding_idx in doc_id_to_graph_idx_mapping.items():
        new_embedding_idx = doc_id_to_new_idx[doc_id]
        if full_graph_embedding_idx is -1:
            embedding_matrix[new_embedding_idx] = average_embedding
        else:
            embedding_matrix[new_embedding_idx] = full_graph_embedding_matrix[full_graph_embedding_idx]
    return embedding_matrix, average_embedding

def load_embedding_matrix(pickle_file,
                          node_to_id_mapping,
                          document_ids,
                          document_id_type = 'scirex'):
    if document_id_type not in ['scirex', 's2orc']:
        raise ValueError("Only 2 possible values supported for document_id_type: scirex or s2orc")
    

    full_graph_embeddings, name2id = unpack_embeddings(pickle_file)
    s2orc_id_to_node_id_mapping = json.load(open(node_to_id_mapping))
    scirex_to_s2orc_mappings = get_scirex_to_s2orc_mappings()

    doc_idx_mapping = determine_document_index_mapping(document_ids)
    doc_id_to_full_matrix_idx_mapping = {}

    missing_documents = 0
    for doc_id in tqdm(document_ids):
        if document_id_type == 'scirex':
            if doc_id in scirex_to_s2orc_mappings:
                converted_doc_id = scirex_to_s2orc_mappings[doc_id]
            else:
                converted_doc_id = None
        else:
            # document id type is s2orc
            converted_doc_id = doc_id

        graph_embedding_idx = -1
        if converted_doc_id is not None and converted_doc_id in s2orc_id_to_node_id_mapping:
            node_id = s2orc_id_to_node_id_mapping[converted_doc_id]
            if str(node_id) in name2id:
                graph_embedding_idx = name2id[str(node_id)]
        if graph_embedding_idx == -1:
            missing_documents += 1
        if document_id_type == 'scirex':
            doc_id_to_full_matrix_idx_mapping[doc_id] = graph_embedding_idx
        else:
            # Document id type is s2orc, therefore we don't need to worry about converted_doc_id being None.
            doc_id_to_full_matrix_idx_mapping[converted_doc_id] = graph_embedding_idx

    print(f"Documents with missing embeddings: {missing_documents}")
    matrix, average_embedding = construct_embedding_matrix(full_graph_embeddings, doc_id_to_full_matrix_idx_mapping, doc_idx_mapping)
    return matrix, doc_idx_mapping, average_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_embeddings_pickle', type=str, required=True, help="GraphVite embedding pickle file")
    parser.add_argument('--node_to_id_mapping', type=str, required=True, help="Node-name-to-node-index JSON file, produced before learning graph embedding")
    parser.add_argument('--out_embedding_file', type=str, required=True, help="Numpy binary file containing document embedding matrix")
    parser.add_argument('--out_doc_to_index_mapping_file', type=str, required=True, help="JSON file containing doc-id-to-embedding-index")
    args = parser.parse_args()
    all_scirex_documents = list(get_scirex_docids())
    matrix, scirex_doc_idx_mapping, _ = load_embedding_matrix(args.graph_embeddings_pickle, args.node_to_id_mapping, all_scirex_documents, document_id_type = 'scirex')

    np.save(args.out_embedding_file, matrix)
    write_scirex_index_mapping_to_file(scirex_doc_idx_mapping, args.out_doc_to_index_mapping_file)
