from collections import defaultdict
import gzip
import json
import jsonlines
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import requests
import time
import wget #       pip install wget


full_data_download_script = "full_data_downloads.sh"
metadata_download_script = "metadata_downloads.sh"
caches_directory = "SciREX/s2orc_caches"

'''
S2ORC information must be accessed in batches, only fetching what you need at each time (due to its massive size).
These functions give data download commands for each shard of S2ORC, which can then be fetched via wget when needed.
'''
def fetch_full_data_download_commands():
    full_data_download_commands = open(full_data_download_script).read().split("\n")
    full_data_download_commands = [x.split() for x in full_data_download_commands if len(x.split()) != 0]
    return full_data_download_commands

def fetch_metadata_download_commands():
    metadata_download_commands = open(metadata_download_script).read().split("\n")
    metadata_download_commands = [x.split() for x in metadata_download_commands if len(x.split()) != 0]
    return metadata_download_commands

class S2OrcEntry:
    def __init__(self, shard_id, doc_id, doc_hash):
        self.shard_id = shard_id
        self.doc_id = doc_id
        self.doc_hash = doc_hash
    
    def __str__(self):
        return str({"shard_id": str(self.shard_id),
                    "doc_id": str(self.doc_id),
                    "doc_hash": str(self.doc_hash)})

class S2Metadata:
    def __init__(self, doc_id, doc_hash, title, doi, arxivId, url):
        self.doc_id = doc_id
        self.doc_hash = doc_hash
        self.title = title
        self.doi = doi
        self.arxivId = arxivId
        self.url = url

    def __str__(self):
        return str({"doc_id": str(self.doc_id),
                    "doc_hash": str(self.doc_hash),
                    "title": str(self.title),
                    "doi": str(self.doi),
                    "arxivId": str(self.arxivId),
                    "url": str(self.url)})


def scirex_document_to_title(doc):
    if len(doc['sections']) == 0:
        raise ValueError("empty document")
    title_tokens = doc['words'][doc['sentences'][0][0]:doc['sentences'][0][1]]
    title = " ".join(title_tokens)
    return title


def get_scirex_docids():
    scirex_train = jsonlines.open("SciREX/scirex_dataset/release_data/train.jsonl")
    scirex_test = jsonlines.open("SciREX/scirex_dataset/release_data/test.jsonl")
    scirex_dev = jsonlines.open("SciREX/scirex_dataset/release_data/dev.jsonl")

    scirex_training_docids = []
    for doc in scirex_train:
        scirex_training_docids.append(doc['doc_id'])
    scirex_test_docids = []
    for doc in scirex_test:
        scirex_test_docids.append(doc['doc_id'])
    scirex_dev_docids = []
    for doc in scirex_dev:
        scirex_dev_docids.append(doc['doc_id'])
    return set(scirex_training_docids + scirex_test_docids + scirex_dev_docids)


def get_shard_id_from_path(shard_path, data_type="metadata"):
    suffix = ".jsonl.gz"
    if data_type == "metadata":
        prefix = "20200705v1/full/metadata/metadata_"
    else:
        prefix = "20200705v1/full/pdf_parses/pdf_parses_"
    if not shard_path.endswith(suffix) or shard_path.find(prefix) == -1:
        raise ValueError(f"Incorrectly formatted path: {shard_path}")
    shard_id = shard_path[shard_path.find(prefix) + len(prefix):-len(suffix)]
    return int(shard_id)


def get_semantic_scholar_metadata(scirex_ids, scirex_s2orc_mappings, overwrite_cache=False):
    s2orc_metadata_cache_file = os.path.join(caches_directory, "s2_metadata.pkl")
    if os.path.exists(s2orc_metadata_cache_file) and not overwrite_cache:
        metadatas = pickle.load(open(s2orc_metadata_cache_file, 'rb'))
    else:
        # Manually hit the SemanticScholar API to get entries (takes about 5 minutes)
        metadatas = {}
        for i, scirex_id in enumerate(scirex_ids):
            if scirex_id in scirex_s2orc_mappings:
                s2orc_id = scirex_s2orc_mappings[scirex_id].doc_id
                if s2orc_id != "":
                    simple_s2_metadata = S2Metadata(str(s2orc_id),
                                    None,
                                    None,
                                    None,
                                    None,
                                    None)
                    metadatas[scirex_id] = simple_s2_metadata
                    continue

            # Remap a few SciREX paper ids known to be bad:
            if scirex_id == "0c278ecf472f42ec1140ca2f1a0a3dd60cbe5c48":
                api_url = f"https://api.semanticscholar.org/v1/paper/9f67b3edc67a35c884bd532a5e73fa3a7f3660d8"
            elif scirex_id == "1a6b67622d04df8e245575bf8fb2066fb6729720":
                api_url = f"https://api.semanticscholar.org/v1/paper/f0ccb215faaeb1e9e86af5827b76c27a8d04e5a7"
            else:
                api_url = f"https://api.semanticscholar.org/v1/paper/{scirex_id}"

            r = requests.get(api_url)
            if r.status_code == 429:
                # Sleep 100 seconds and retry, in case we've hit the service rate limiter
                while True:
                    print("Too many requests error!")
                    time.Sleep(100)
                    r = requests.get(api_url)
                    if r.status_code != 429:
                        break
            #
            response = r.json()        
            # Verify that all fields are in response
            keys = ["corpusId", "paperId", "title", "doi", "arxivId", "url"]
            for k in keys:
                if k not in response:
                    print(f"SciREX document {i} missing key {k}")
                    break
            #
            s2_metadata = S2Metadata(str(response.get("corpusId")),
                                    response.get("paperId"),
                                    response.get("title"),
                                    response.get("doi"),
                                    response.get("arxivId"),
                                    response.get("url"))
            metadatas[scirex_id] = s2_metadata
            if (i+1) % 10 == 0:
                print(f"{i+1} document metadatas downloaded")
            # Rate-limit
            time.sleep(3)
        pickle.dump(metadatas, open(s2orc_metadata_cache_file, 'wb'))

    return metadatas

def fetch_s2orc_keys_from_scirex_ids(scirex_doc_ids, data_download_commands):
    scirex_s2orc_mapping_file = os.path.join(caches_directory, "s2orc_hash_to_struct_mapping.pkl")
    if os.path.exists(scirex_s2orc_mapping_file):
        s2orc_hash_to_struct_mapping = pickle.load(open(scirex_s2orc_mapping_file, 'rb'))
    else:
        # If we don't have a cache file already, then manually match s2orc and scirex entries
        # (takes several hours and requires downloading and purging hundreds of GB of data)
        s2orc_hash_to_struct_mapping = {}
        start = time.perf_counter()
        for i, s2orc_shard_command in enumerate(data_download_commands):
            output_path = f"s2orc_downloads/{s2orc_shard_command[2]}"
            data_url = eval(s2orc_shard_command[3])
            shard_id = get_shard_id_from_path(output_path, data_type="full_text")
            #
            wget.download(data_url, out=output_path)
            #
            end = time.perf_counter()
            print(f"Took {end - start} seconds to download shard {i}\n")
            start = end
            #
            shard = gzip.open(output_path, 'rt')
            s2orc_full = jsonlines.Reader(shard)

            hits = 0
            for doc in s2orc_full:
                doc_hash = doc["_pdf_hash"]
                if doc['_pdf_hash'] in scirex_doc_ids:
                    doc_id = doc["paper_id"]
                    s2orc_hash_to_struct_mapping[doc_hash] = S2OrcEntry(shard_id, doc_id, doc_hash)
                    hits += 1
            #
            print(f"{hits} matching documents found!")

            end = time.perf_counter()
            print(f"Took {end - start} seconds to process pdf parses")
            start = end
            #
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"Deleted {output_path}")
            print("\n")
        pickle.dump(s2orc_hash_to_struct_mapping, open("s2orc_hash_to_struct_mapping.pkl", 'wb'))
    return s2orc_hash_to_struct_mapping


def s2orc_document_matches_s2_metadata(s2orc_doc, scirex_metadata, scirex_id):
    if scirex_metadata.doc_id is not None and scirex_metadata.doc_id is not "":
        # TODO: un-comment this code, if there's an indication that the S2ORC data schema is not
        # consistently typed
        #
        # if not isinstance(s2orc_doc["paper_id"], str):
        #    s2orc_doc["paper_id"] = str(s2orc_doc["paper_id"])
        if scirex_metadata.doc_id == s2orc_doc["paper_id"]:
            print(f">> SciREX document {scirex_id} matched on paper ID: {scirex_metadata.doc_id}")
            return True

    if scirex_metadata.doi is not None and scirex_metadata.doi is not "":
        if scirex_metadata.doi == s2orc_doc["doi"]:
            print(f">> SciREX document {scirex_id} matched on DOI: {scirex_metadata.doi}")
            return True

    if scirex_metadata.arxivId is not None and scirex_metadata.arxivId is not "":
        if scirex_metadata.arxivId == s2orc_doc["arxiv_id"]:
            print(f">> SciREX document {scirex_id} matched on arxivId: {scirex_metadata.arxivId}")
            return True

    # Disabling URL filtering, since S2ORC uses the CorpusID url scheme while SciREX uses doc hash.
    #
    #if s2_metadata.url is not None and s2_metadata.url is not "":
    #    if s2_metadata.url == s2_metadata["s2_url"]:
    #        print(f">> SciREX document {scirex_id} matched on S2 URL: {s2_metadata.url}")
    #        return True

    # TODO: Try doing a fuzzy title match (e.g. edit distance) instead of exact string match
    if scirex_metadata.title is not None and scirex_metadata.title is not "":
        if scirex_metadata.title == s2orc_doc["title"]:
            print(f">> SciREX document {scirex_id} matched on arxivId: {scirex_metadata.title}")
            return True

    return False


def load_metadata_into_dicts(scirex_documents_metadata):
    doc_id_dict = {}
    doi_dict = {}
    arxiv_id_dict = {}
    title_dict = {}

    for scirex_id, doc_meta in scirex_documents_metadata.items():
        if scirex_id is None:
            continue

        if doc_meta.doc_id is not None and doc_meta.doc_id is not "":
            doc_id_dict[doc_meta.doc_id] = scirex_id
        if doc_meta.doi is not None and doc_meta.doi is not "":
            doi_dict[doc_meta.doi] = scirex_id
        if doc_meta.arxivId is not None and doc_meta.arxivId is not "":
            arxiv_id_dict[doc_meta.arxivId] = scirex_id
        if doc_meta.title is not None and doc_meta.title is not "":
            title_dict[doc_meta.title] = scirex_id

    return [doc_id_dict, doi_dict, arxiv_id_dict, title_dict]

def delete_keys_with_value(fdict, value):
    if value not in fdict.values():
        return
    matching_keys = [k for (k, v) in fdict.items() if v == value]
    for k in matching_keys:
        del fdict[k]

def query_s2orc_document_in_scirex_dict(s2orc_doc, field_dicts):
    s2orc_paper_id = s2orc_doc.get("paper_id")
    if s2orc_paper_id is not None:
        scirex_id = field_dicts[0].get(s2orc_paper_id)
        if scirex_id is not None:
            print(f">> SciREX document {scirex_id} matched on paper ID: {s2orc_paper_id}")
            for fdict in field_dicts:
                delete_keys_with_value(fdict, scirex_id)
            return scirex_id

    s2orc_doi = s2orc_doc.get("doi")
    if s2orc_doi is not None:
        scirex_id = field_dicts[1].get(s2orc_doi)
        if scirex_id is not None:
            print(f">> SciREX document {scirex_id} matched on DOI: {s2orc_doi}")
            for fdict in field_dicts:
                delete_keys_with_value(fdict, scirex_id)
            return scirex_id

    s2orc_arxiv_id = s2orc_doc.get("arxiv_id")
    if s2orc_arxiv_id is not None:
        scirex_id = field_dicts[2].get(s2orc_arxiv_id)
        if scirex_id is not None:
            print(f">> SciREX document {scirex_id} matched on arxiv ID: {s2orc_arxiv_id}")
            for fdict in field_dicts:
                delete_keys_with_value(fdict, scirex_id)
            return scirex_id

    s2orc_title = s2orc_doc.get("title")
    if s2orc_title is not None:
        scirex_id = field_dicts[3].get(s2orc_title)
        if scirex_id is not None:
            print(f">> SciREX document {scirex_id} matched on paper title: {s2orc_title}")
            for fdict in field_dicts:
                delete_keys_with_value(fdict, scirex_id)
            return scirex_id

    return None


# Using Semantic scholar metadata, use available identifiers to link with S2ORC meta fields
# After linking, save full s2 meta objects in mapping from
def fetch_s2orc_meta_rows_from_scirex_ids(scirex_document_metadata, metadata_download_commands, overwrite_cache=False, num_shards_to_use=None):
    function_start = time.perf_counter()
    all_hits = 0

    field_dicts = load_metadata_into_dicts(scirex_document_metadata)

    scirex_s2orc_metadata_file = os.path.join(caches_directory, "scirex_id_to_s2orc_metadata.pkl")
    if os.path.exists(scirex_s2orc_metadata_file) and not overwrite_cache:
        scirex_s2orc_metadata = pickle.load(open(scirex_s2orc_metadata_file, 'rb'))
    else:
        # If we don't have a cache file already, then manually match s2orc and scirex entries
        # (takes several hours and requires downloading and purging tens of GB of data)
        scirex_s2orc_metadata = {}
        start = time.perf_counter()
        for i, s2orc_shard_command in enumerate(metadata_download_commands):
            output_path = f"s2orc_downloads/{s2orc_shard_command[2]}"
            data_url = eval(s2orc_shard_command[3])
            shard_id = get_shard_id_from_path(output_path, data_type="metadata")
            print(f"Starting processing of shard {shard_id}")
            start = time.perf_counter()
            if not os.path.exists(output_path):
                wget.download(data_url, out=output_path)
            # Time download
            end = time.perf_counter()
            print(f"Took {end - start} seconds to download shard {shard_id}\n")
            start = end
            # Load and match metadata rows
            shard = gzip.open(output_path, 'rt')
            s2orc_metadata = jsonlines.Reader(shard)
            hits = 0
            for doc in s2orc_metadata:
                scirex_hit = query_s2orc_document_in_scirex_dict(doc, field_dicts)
                if scirex_hit is not None:
                    hits += 1
                    doc["shard_id"] = shard_id
                    scirex_s2orc_metadata[scirex_hit] = doc

            print(f"{hits} matching documents found in shard {shard_id}")
            all_hits += hits

            end = time.perf_counter()
            print(f"Took {end - start} seconds to process pdf parses")

            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"Deleted {output_path}")
            print("\n")
            if num_shards_to_use is not None and i + 1 >= num_shards_to_use:
                break
        print(f"Found {all_hits} total matching documents in {time.perf_counter() - function_start} seconds")
        pickle.dump(scirex_s2orc_metadata, open(scirex_s2orc_metadata_file, 'wb'))
    return scirex_s2orc_metadata


def create_citation_graph_from_seed_nodes(seed_node_ids, metadata_download_commands, graph_radius = 2, overwrite_cache=False, num_shards_to_use=None):
    function_start = time.perf_counter()

    citation_graph_file = os.path.join(caches_directory, f"citation_graph_radius_{graph_radius}.pkl")
    if os.path.exists(citation_graph_file) and not overwrite_cache:
        citation_graph_adjacency_lists = pickle.load(open(citation_graph_file, 'rb'))
        print(f"Loading citation graph of radius {graph_radius} took {time.perf_counter() - function_start} seconds")
    else:
        out_edges = {}
        in_edges = {}
        for radius in range(1, graph_radius + 1):
            seed_node_ids = set(seed_node_ids)
            citation_graph_file = os.path.join(caches_directory, f"citation_graph_radius_{radius}.pkl")
            if os.path.exists(citation_graph_file) and not overwrite_cache:
                citation_graph_adjacency_lists = pickle.load(open(citation_graph_file, 'rb'))
                (out_edges, in_edges) = citation_graph_adjacency_lists
                cited_nodes = [out_id for out_neighbors in out_edges.values() for out_id in out_neighbors]
                citing_nodes = [in_id for in_neighbors in in_edges.values() for in_id in in_neighbors ]
                seed_node_ids = set(cited_nodes + citing_nodes)
                print(f"Loaded round {radius}:")
                continue

            print(f"Starting round {radius}:")

            # If we don't have a cache file already, then manually match s2orc and scirex entries
            # (takes several hours and requires downloading and purging tens of GB of data)
            for i, s2orc_shard_command in enumerate(metadata_download_commands):
                output_path = f"s2orc_downloads/{s2orc_shard_command[2]}"
                data_url = eval(s2orc_shard_command[3])
                shard_id = get_shard_id_from_path(output_path, data_type="metadata")
                print(f"Starting processing of shard {shard_id}")
                start = time.perf_counter()
                if not os.path.exists(output_path):
                    wget.download(data_url, out=output_path)
                # Time download
                end = time.perf_counter()
                print(f"Took {end - start} seconds to download shard {shard_id}\n")
                start = end
                # Load and match metadata rows
                shard = gzip.open(output_path, 'rt')
                s2orc_metadata = jsonlines.Reader(shard)
                hits = 0
                nodes_added = 0
                for doc in s2orc_metadata:
                    paper_id = doc['paper_id']
                    if paper_id not in seed_node_ids:
                        # Only looking to add papers adjacent to our query nodes.
                        continue
                    if paper_id in out_edges or paper_id in in_edges:
                        # No need to re-add papers already in the adjacency list
                        continue
                    hits += 1
                    if 'outbound_citations' in doc and len(doc['outbound_citations']) > 0:
                        out_edges[paper_id] = doc['outbound_citations']
                        nodes_added += len(doc['outbound_citations'])
                    if 'inbound_citations' in doc and len(doc['inbound_citations']) > 0:
                        in_edges[paper_id] = doc['inbound_citations']
                        nodes_added += len(doc['inbound_citations'])

                end = time.perf_counter()
                print(f"{nodes_added} new documents added from graph search in shard {shard_id}, from {hits} query nodes")
                print(f"Took {end - start} seconds to add nodes for shard {shard_id}")

                if os.path.exists(output_path):
                    os.remove(output_path)
                print(f"Deleted {output_path}")
                print("\n")
                if num_shards_to_use is not None and i + 1 >= num_shards_to_use:
                    break

            cited_nodes = [out_id for out_neighbors in out_edges.values() for out_id in out_neighbors]
            citing_nodes = [in_id for in_neighbors in in_edges.values() for in_id in in_neighbors ]
            seed_node_ids = set(cited_nodes + citing_nodes)
            print(f"\n\nCreated citation graph of radius {radius} from SciREX documents. Contains {len(seed_node_ids)} documents.\n\n")
            citation_graph_adjacency_lists = (out_edges, in_edges)
            pickle.dump(citation_graph_adjacency_lists, open(citation_graph_file, 'wb'))
            print(f"Creating citation graph of radius {graph_radius} took {time.perf_counter() - function_start} seconds")

    return citation_graph_adjacency_lists


def construct_neighbor_text(seed_node_ids, metadata_download_commands, full_text_download_commands, out_directory = "SciREX/s2orc_caches/fulltexts", overwrite_cache=False, num_shards_to_use=None):
    function_start = time.perf_counter()
    scigraph_documents_path = os.path.join(out_directory, "scigraph_full_documents.jsonl")
    doc_to_index_path = os.path.join(out_directory, "doc_to_line_idxs.json")
    if os.path.exists(doc_to_index_path) and os.path.exists(doc_to_index_path):
        doc_idx_mapping = json.load(open(doc_to_index_path))
    else:
        scigraph_documents_file = open(scigraph_documents_path, 'wb')
        writer = jsonlines.Writer(scigraph_documents_file)

        (out_edges, in_edges) = create_citation_graph_from_seed_nodes(seed_node_ids, metadata_download_commands, graph_radius=1, overwrite_cache=overwrite_cache, num_shards_to_use=num_shards_to_use)
        all_neighbors = [v for to_nodes in out_edges.values() for v in to_nodes] + [v for from_nodes in in_edges.values() for v in from_nodes]
        all_neighbors = set(all_neighbors)
        doc_idx_mapping = {}

        for i, s2orc_shard_command in enumerate(full_text_download_commands):
            output_path = f"s2orc_downloads/{s2orc_shard_command[2]}"
            data_url = eval(s2orc_shard_command[3])
            shard_id = get_shard_id_from_path(output_path, data_type="full_text")
            print(f"Starting processing of shard {shard_id}")
            start = time.perf_counter()
            if not os.path.exists(output_path):
                wget.download(data_url, out=output_path)
            # Time download
            end = time.perf_counter()
            print(f"Took {end - start} seconds to download shard {shard_id}\n")
            start = end
            # Load and match metadata rows
            shard = gzip.open(output_path, 'rt')
            s2orc_metadata = jsonlines.Reader(shard)
            hits = 0
            nodes_added = 0
            for doc in s2orc_metadata:
                paper_id = doc['paper_id']
                if paper_id in all_neighbors:
                    writer.write(doc)
                    doc_idx_mapping[paper_id] = len(doc_idx_mapping)
                    all_neighbors.remove(paper_id)

            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"Deleted {output_path}")
            print("\n")
            if num_shards_to_use is not None and i + 1 >= num_shards_to_use:
                break

        writer.close()
        print(f"Wrote {len(doc_idx_mapping)} jsonlines to {scigraph_documents_path}")
        json.dump(doc_idx_mapping, open(doc_to_index_path, 'w'))

    elapsed = time.perf_counter() - function_start
    print(f"Constructing neighbor texts took {elapsed} seconds")
    return doc_idx_mapping, scigraph_documents_path

def get_scirex_s2_metadata():
    scirex_paper_ids = list(get_scirex_docids())
    metadata_download_commands = fetch_metadata_download_commands()
    full_data_download_commands = fetch_full_data_download_commands()

    s2orc_hash_to_struct_mapping = fetch_s2orc_keys_from_scirex_ids(scirex_paper_ids, full_data_download_commands)
    scirex_s2_metadata = get_semantic_scholar_metadata(scirex_paper_ids, s2orc_hash_to_struct_mapping)
    scirex_s2orc_metadata = fetch_s2orc_meta_rows_from_scirex_ids(scirex_s2_metadata, metadata_download_commands)  
    return scirex_s2orc_metadata

def get_s2orc_scirex_id_mapping():
    scirex_s2orc_metadata = get_scirex_s2_metadata()
    s2orc_to_scirex_mappings = {}
    for scirex, s2orc in scirex_s2orc_metadata.items():
        s2orc_to_scirex_mappings[s2orc["paper_id"]] = scirex
    return s2orc_to_scirex_mappings

def get_scirex_to_s2orc_mappings():
    scirex_s2orc_metadata = get_scirex_s2_metadata()
    scirex_to_s2orc_mappings = {}
    for scirex, s2orc in scirex_s2orc_metadata.items():
        scirex_to_s2orc_mappings[scirex] = s2orc["paper_id"]
    return scirex_to_s2orc_mappings


def get_citation_graph(radius, remap_to_scirex_id = False):
    scirex_paper_ids = list(get_scirex_docids())

    metadata_download_commands = fetch_metadata_download_commands()
    s2orc_to_scirex_mappings = get_s2orc_scirex_id_mapping()
    (out_edges, in_edges) = create_citation_graph_from_seed_nodes(scirex_paper_ids, metadata_download_commands, graph_radius = radius)
    out_edges_scirex_keys = {}
    in_edges_scirex_keys = {}
    for k, v in out_edges.items():
        if remap_to_scirex_id:
            if k in s2orc_to_scirex_mappings:
                out_edges_scirex_keys[s2orc_to_scirex_mappings[k]] = v
        else:
            out_edges_scirex_keys[k] = v
    for k, v in in_edges.items():
        if remap_to_scirex_id:
            if k in s2orc_to_scirex_mappings:
                in_edges_scirex_keys[s2orc_to_scirex_mappings[k]] = v
        else:
            in_edges_scirex_keys[k] = v

    return out_edges_scirex_keys, in_edges_scirex_keys


def compute_scirex_documents_graph_degrees(remap_to_scirex_id = True, degree_direction="both"):
    assert degree_direction in ["both", "out", "in"]

    out_graph, in_graph = get_citation_graph(1, remap_to_scirex_id=remap_to_scirex_id)
    scirex_paper_ids = list(get_scirex_docids())
    degrees = {}
    for doc_id in scirex_paper_ids:
        degrees[doc_id] = 0
        if degree_direction == "both" or degree_direction == "in":
            degrees[doc_id] += len(in_graph.get(doc_id, []))
        if degree_direction == "both" or degree_direction == "out":
            degrees[doc_id] += len(out_graph.get(doc_id, []))
    return degrees


def draw_bar_chart(data, ylabel=None, xlabel=None, fname="/tmp/scratch.png", num_buckets=100, xlimit=None):
    fig, ax = plt.subplots()
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlimit:
        ax.set_xlim(0, xlimit)
    n, bins, patches = ax.hist(data, num_buckets, color="lightblue", rwidth=0.8)
    print(f"Wrote figure to {fname}")
    fig.savefig(fname, dpi=400, bbox_inches='tight')
    plt.tight_layout()
    del fig

def plot_histogram_of_degrees():
    out_graph, in_graph = get_citation_graph(1, remap_to_scirex_id=True)
    scirex_paper_ids = list(get_scirex_docids())
    in_degrees = {}
    out_degrees={}
    for doc_id in scirex_paper_ids:
        in_degrees[doc_id] = 0
        out_degrees[doc_id] = 0
        in_degrees[doc_id] += len(in_graph.get(doc_id, []))
        out_degrees[doc_id] += len(out_graph.get(doc_id, []))

    in_degree_counts = list(in_degrees.values())
    out_degree_counts = list(out_degrees.values())

    plt.rcParams["font.size"] = "30"
    draw_bar_chart(out_degree_counts, ylabel=None, xlabel="Out-degree", fname="/tmp/out_degree_histogram.pdf", num_buckets=30)
    draw_bar_chart(in_degree_counts, ylabel="Frequency", xlabel="In-degree", fname="/tmp/in_degree_histogram.pdf", xlimit=2500, num_buckets=350)

def bucket_documents_by_graph_degree(test_set, num_buckets=6, degree_direction="both", remap_to_scirex_id = True):
    all_degrees = compute_scirex_documents_graph_degrees(remap_to_scirex_id=remap_to_scirex_id, degree_direction=degree_direction)
    test_degrees = {}
    for doc in test_set:
        test_degrees[doc] = all_degrees.get(doc, 0)

    docs_sorted_by_degree = list(sorted(test_degrees.items(), key=lambda x:x[1]))
    bucket_size = np.ceil(float(len(docs_sorted_by_degree))/num_buckets)
    buckets = [((0,0), [])]
    for doc, degree in docs_sorted_by_degree:
        if len(buckets[-1][1]) >= bucket_size:
            buckets.append(((buckets[-1][0][1], 0), []))

        ((start, end), docs) = buckets[-1]
        docs.append(doc)
        if degree > end:
            end = degree
        buckets[-1] = ((start, end), docs)
    return buckets

def bucket_documents_by_manual_buckets(test_set, buckets, degree_direction="both", remap_to_scirex_id = True):
    all_degrees = compute_scirex_documents_graph_degrees(remap_to_scirex_id=remap_to_scirex_id, degree_direction=degree_direction)
    buckets = [(bucket, []) for bucket in buckets]
    for doc in test_set:
        graph_degree = all_degrees.get(doc, 0)
        for i in range(len(buckets)):
            bucket_start = buckets[i][0][0]
            bucket_end = buckets[i][0][1]
            if graph_degree >= bucket_start and (graph_degree < bucket_end or (graph_degree <= bucket_end and i == len(buckets) - 1)):
                buckets[i][1].append(doc)
                break
    assert set([x for y in buckets for x in y[1]]) == set(test_set), breakpoint()
    return buckets




def bucket_documents_by_manual_buckets(test_set, buckets, degree_direction="both", remap_to_scirex_id = True):
    all_degrees = compute_scirex_documents_graph_degrees(remap_to_scirex_id=remap_to_scirex_id, degree_direction=degree_direction)
    buckets = [(bucket, []) for bucket in buckets]
    for doc in test_set:
        graph_degree = all_degrees.get(doc, 0)
        for i in range(len(buckets)):
            bucket_start = buckets[i][0][0]
            bucket_end = buckets[i][0][1]
            if graph_degree >= bucket_start and (graph_degree < bucket_end or (graph_degree <= bucket_end and i == len(buckets) - 1)):
                buckets[i][1].append(doc)
                break
    assert set([x for y in buckets for x in y[1]]) == set(test_set), breakpoint()
    return buckets


def get_scirex_neighbor_texts():
    scirex_paper_ids = list(get_scirex_docids())
    full_data_download_commands = fetch_full_data_download_commands()
    metadata_download_commands = fetch_metadata_download_commands()

    doc_idx_mapping, scigraph_documents_path = construct_neighbor_text(scirex_paper_ids, metadata_download_commands, full_data_download_commands, overwrite_cache=False)
    return doc_idx_mapping, scigraph_documents_path


def construct_graphvite_graph_format(citation_graph,
                                     train_ratio=0.95,
                                     test_ratio=0.025,
                                     val_ratio=0.025,
                                     graph_edges_file_prefix="s2orc",
                                     node_mapping_file="node_id_mapping.json",
                                     make_undirected=True):
    function_start = time.perf_counter()
    graph_edges = []
    (citing_graph, cited_graph) = citation_graph
    node_mapping = {}

    mapped_edges = defaultdict(list)
    for graph in [citing_graph, cited_graph]:
        for src, dsts in graph.items():
            if src in node_mapping:
                src_num = node_mapping[src]
            else:
                src_num = len(node_mapping) + 1
                node_mapping[src] = src_num

            for dst in dsts:
                if dst in node_mapping:
                    dst_num = node_mapping[dst]
                else:
                    dst_num = len(node_mapping) + 1
                    node_mapping[dst] = dst_num

                if make_undirected and src_num > dst_num:
                    # Make sure src idx is always less than dst, to ensure each edge appears
                    # only once.
                    graph_edges.append(f"{dst_num}\t{src_num}")
                    mapped_edges[dst_num].append(src_num)
                else:
                    graph_edges.append(f"{src_num}\t{dst_num}")
                    mapped_edges[src_num].append(dst_num)

    json.dump(node_mapping, open(node_mapping_file, 'w'))

    graph_edges_unique = list(set(graph_edges))
    random.shuffle(graph_edges_unique)
    train_size = int(train_ratio * len(graph_edges_unique))
    test_size = int(test_ratio * len(graph_edges_unique))
    train_data = graph_edges_unique[:train_size]
    test_data = graph_edges_unique[train_size:train_size+test_size]
    val_data = graph_edges_unique[train_size+test_size:]
    train_data = sorted(train_data)
    # Produce special test data for link prediction evaluation
    test_data = [f"{t}\t1" for t in test_data]

    all_node_ids = list(node_mapping.values())
    # Generating negative edges
    print("Generating negative edges")
    num_negative_edges_generated = 0
    start = time.perf_counter()
    for i in range(len(test_data) * 5):
        [src] = random.sample(all_node_ids, k=1)
        while True:
            [trg] = random.sample(all_node_ids, k=1)
            if trg not in mapped_edges[src]:
                break
        test_data.append(f"{src}\t{trg}\t0")
        num_negative_edges_generated += 1
    end = time.perf_counter()
    print(f"Done generating {num_negative_edges_generated} negative edges: took {end - start} seconds")

    test_data = sorted(test_data)
    val_data = sorted(val_data)

    splits = {"train": train_data, "test": test_data, "val": val_data}

    for split_name, split in splits.items():
        fname = f"{graph_edges_file_prefix}_{split_name}.txt"
        f = open(fname, 'w')
        f.write("\n".join(split))
        print(f"Wrote {len(split)} edges to {fname}.")
        f.close()

    directedness = "undirected" if make_undirected else "directed"
    print(f"Creating {directedness} edge lists took {time.perf_counter() - function_start} seconds.")

def main():
    all_scirex_docids = get_scirex_docids()

    # This must be obtained by requesting access from the S2ORC team (https://allenai.org/data/s2orc)
    full_data_download_commands = fetch_full_data_download_commands()
    print("Load existing SciREX-to-S2ORC mappings:")
    s2orc_hash_to_struct_mapping = fetch_s2orc_keys_from_scirex_ids(all_scirex_docids, full_data_download_commands)
    print("Done.\n\n")

    print("Get SciREX document metadata from Semantic Scholar API:")
    scirex_s2_metadata = get_semantic_scholar_metadata(all_scirex_docids, s2orc_hash_to_struct_mapping)
    print("Done.\n\n")

    print("Link SciREX documents with S2ORC documents")
    metadata_download_commands = fetch_metadata_download_commands()
    scirex_s2orc_metadata = fetch_s2orc_meta_rows_from_scirex_ids(scirex_s2_metadata, metadata_download_commands)

    scirex_paper_ids = [doc['paper_id'] for doc in scirex_s2orc_metadata.values()]
    print(f"Constructing neighbor texts")
    _ = construct_neighbor_text(scirex_paper_ids, metadata_download_commands, full_data_download_commands, overwrite_cache=False)
    '''
    citation_graph = create_citation_graph_from_seed_nodes(scirex_paper_ids, metadata_download_commands, graph_radius=2, overwrite_cache=False)

    construct_graphvite_graph_format(citation_graph)
    '''
    print("Totally Done.")


if __name__ == "__main__":
    main()
