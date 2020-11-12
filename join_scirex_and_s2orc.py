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


all_scirex_docids = get_scirex_docids()


data_download_script = "full_data_downloads.sh"
data_download_commands = open("full_data_downloads.sh").read().split("\n")
data_download_commands = [x.split() for x in data_download_commands if len(x.split()) != 0]


def get_shard_id_from_path(shard_path):
    suffix = ".jsonl.gz"
    prefix = "20200705v1/full/pdf_parses/pdf_parses_"
    if not shard_path.endswith(suffix) or shard_path.find(prefix) == -1:
        raise ValueError(f"Incorrectly formatted path: {shard_path}")
    shard_id = shard_path[shard_path.find(prefix) + len(prefix):-len(suffix)]
    return int(shard_id)


def get_s2orc_ids(scirex_ids):
    s2orc_metadata_cache_file = "s2_metadata.pkl"
    if os.path.exists(s2orc_metadata_cache):
        metadatas = pickle.load(open(s2orc_metadata_cache_file, 'rb'))
    else:
        # Manually hit the SemanticScholar API to get entries (takes about 5 minutes)
        metadatas = {}
        for i, scirex_id in enumerate(scirex_ids):
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


def main():
    start = time.perf_counter()

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
            shard_id = get_shard_id_from_path(output_path)
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
                if doc['_pdf_hash'] in all_scirex_docids:
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

    s2meta = get_s2orc_ids(mismatched_docs)


if __name__ == "__main__":
    main()
