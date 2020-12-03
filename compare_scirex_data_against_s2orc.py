# Deprecated
"""
import gzip
import jsonlines
import os
import pickle
import requests
import time
import wget #       pip install wget

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
    scirex_train = jsonlines.open("/projects/ogma1/vijayv/SciREX/scirex_dataset/release_data/train.jsonl")
    scirex_test = jsonlines.open("/projects/ogma1/vijayv/SciREX/scirex_dataset/release_data/test.jsonl")
    scirex_dev = jsonlines.open("/projects/ogma1/vijayv/SciREX/scirex_dataset/release_data/dev.jsonl")

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
    s2orc_metadata_cache_file = "s2_metadata.pkl"
    if os.path.exists(s2orc_metadata_cache_file) and not overwrite_cache:
        metadatas = pickle.load(open(s2orc_metadata_cache_file, 'rb'))
    else:
        # Manually hit the SemanticScholar API to get entries (takes about 5 minutes)
        metadatas = {}
        for i, scirex_id in enumerate(scirex_ids):
            if scirex_id in scirex_s2orc_mappings:
                s2orc_id = scirex_s2orc_mappings[scirex_id].doc_id
                if s2orc_id != "":
                    simple_s2_metadata = S2Metadata(s2orc_id,
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
            s2_metadata = S2Metadata(response.get("corpusId"),
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
    scirex_s2orc_mapping_file = "s2orc_hash_to_struct_mapping.pkl"
    if os.path.exists(scirex_s2orc_mapping_file):
        s2orc_hash_to_struct_mapping = pickle.load(open("s2orc_hash_to_struct_mapping.pkl", 'rb'))
    else:
        # If we don't have a cache file already, then manually match s2orc and scirex entries
        # (takes several hours and requires downloading and purging hundreds of GB of data)
        s2orc_hash_to_struct_mapping = {}
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


def s2orc_document_matches_s2_metadata(s2orc_doc, s2_metadata, scirex_id):
    if s2_metadata.doc_id is not None and s2_metadata.doc_id is not "":
        if s2_metadata.doc_id == s2_metadata["paper_id"]:
            print(f">> SciREX document {scirex_id} matched on paper ID: {s2_metadata.doc_id}")
            return True
    if s2_metadata.doi is not None and s2_metadata.doi is not "":
        if s2_metadata.doi == s2_metadata["doi"]:
            print(f">> SciREX document {scirex_id} matched on DOI: {s2_metadata.doi}")
            return True

    if s2_metadata.arxivId is not None and s2_metadata.arxivId is not "":
        if s2_metadata.arxivId == s2_metadata["arxiv_id"]:
            print(f">> SciREX document {scirex_id} matched on arxivId: {s2_metadata.arxivId}")
            return True

    # Disabling URL filtering, since S2ORC uses the CorpusID url scheme while SciREX uses doc hash.
    #
    #if s2_metadata.url is not None and s2_metadata.url is not "":
    #    if s2_metadata.url == s2_metadata["s2_url"]:
    #        print(f">> SciREX document {scirex_id} matched on S2 URL: {s2_metadata.url}")
    #        return True

    # TODO: Try doing a fuzzy title match (e.g. edit distance) instead of exact string match
    if s2_metadata.title is not None and s2_metadata.title is not "":
        if s2_metadata.url == s2_metadata["title"]:
            return True

    return False


# Using Semantic scholar metadata, use available identifiers to link with S2ORC meta fields
# After linking, save full s2 meta objects in mapping from 
def fetch_s2orc_meta_rows_from_scirex_ids(scirex_document_metadata, metadata_download_commands, overwrite_cache=False, num_shards_to_use=None):
    function_start = time.perf_counter()

    scirex_s2orc_metadata_file = "scirex_id_to_s2orc_metadata.pkl"
    if os.path.exists(scirex_s2orc_metadata_file) and not overwrite_cache:
        scirex_s2orc_metadata = pickle.load(open(scirex_s2orc_metadata_file, 'rb'))
    else:
        # If we don't have a cache file already, then manually match s2orc and scirex entries
        # (takes several hours and requires downloading and purging tens of GB of data)
        scirex_s2orc_metadata = {}
        for i, s2orc_shard_command in enumerate(metadata_download_commands):
            output_path = f"s2orc_downloads/{s2orc_shard_command[2]}"
            data_url = eval(s2orc_shard_command[3])
            shard_id = get_shard_id_from_path(output_path, data_type="metadata")
            start = time.perf_counter()
            if not os.path.exists(output_path):
                wget.download(data_url, out=output_path)
            # Time download
            end = time.perf_counter()
            print(f"Took {end - start} seconds to download shard {i}\n")
            start = end
            # Load and match metadata rows
            shard = gzip.open(output_path, 'rt')
            s2orc_metadata = jsonlines.Reader(shard)
            hits = 0
            for doc in s2orc_metadata:
                for scirex_id, s2_metadata_entry in scirex_document_metadata.items():
                    if s2orc_document_matches_s2_metadata(doc, s2_metadata_entry, scirex_id):
                        scirex_s2orc_metadata[scirex_id] = doc
                        hits += 1
            #
            print(f"{hits} matching documents found in shard {shard_id}")

            end = time.perf_counter()
            print(f"Took {end - start} seconds to process pdf parses")
            #
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"Deleted {output_path}")
            print("\n")
            if num_shards_to_use is not None and i + 1 >= num_shards_to_use:
                break

        pickle.dump(scirex_s2orc_metadata, open(scirex_s2orc_metadata_file, 'wb'))
    
    print(f"Found {hits} total matching documents in {time.perf_counter() - function_start} seconds")
    return scirex_s2orc_metadata


def main():
    all_scirex_docids = get_scirex_docids()

    # This must be obtained by requesting access from the S2ORC team (https://allenai.org/data/s2orc)
    full_data_download_script = "full_data_downloads.sh"
    full_data_download_commands = open(full_data_download_script).read().split("\n")
    full_data_download_commands = [x.split() for x in full_data_download_commands if len(x.split()) != 0]
    s2orc_hash_to_struct_mapping = fetch_s2orc_keys_from_scirex_ids(all_scirex_docids, full_data_download_commands)

    scirex_s2_metadata = get_semantic_scholar_metadata(all_scirex_docids, s2orc_hash_to_struct_mapping)

    metadata_download_script = "metadata_downloads.sh"
    metadata_download_commands = open(metadata_download_script).read().split("\n")
    metadata_download_commands = [x.split() for x in metadata_download_commands if len(x.split()) != 0]
    scirex_s2orc_metadata = fetch_s2orc_meta_rows_from_scirex_ids(scirex_s2_metadata, metadata_download_commands, num_shards_to_use=1)



if __name__ == "__main__":
    main()
"""