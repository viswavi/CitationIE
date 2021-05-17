from collections import defaultdict
import gzip
import json
import jsonlines
import os
import pickle
import shutil
import time
import wget

from join_scirex_and_s2orc import (
                                    S2OrcEntry,
                                    S2Metadata,
                                    fetch_full_data_download_commands,
                                    fetch_metadata_download_commands,
                                    caches_directory,
                                    get_shard_id_from_path
                                  )


def download_s2orc_data(download_commands, candidate_set, data_directory, data_type, num_shards_to_use=None):
    shard_index = defaultdict(list)
    for i, s2orc_shard_command in enumerate(download_commands):
        output_path = f"s2orc_downloads/{s2orc_shard_command[2]}"
        data_url = eval(s2orc_shard_command[3])
        shard_id = get_shard_id_from_path(output_path, data_type=data_type)
        print(f"Starting processing of shard {shard_id}")
        start = time.perf_counter()
        if not os.path.exists(output_path):
            wget.download(data_url, out=output_path)
        # Time download
        end = time.perf_counter()
        print(f"\nTook {round(end - start, 3)} seconds to download shard {shard_id}.")
        start = end

        shard_file_name = os.path.join(data_directory, f"{shard_id}.jsonl")
        shard_writer = jsonlines.open(shard_file_name, 'w')

        # Load and match metadata rows
        shard = gzip.open(output_path, 'rt')
        s2orc_metadata = jsonlines.Reader(shard)
        hits = 0
        for doc in s2orc_metadata:
            paper_id = doc['paper_id']
            if paper_id in candidate_set:
                shard_writer.write(doc)
                shard_index[shard_id].append(paper_id)
                hits += 1

        end = time.perf_counter()

        print(f"Wrote {hits} hits from shard {shard_id} to {shard_file_name}.")
        print(f"Processing docs in shard took {end - start} seconds.")

        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"Deleted {output_path}")
        print("\n")
        if num_shards_to_use is not None and i + 1 >= num_shards_to_use:
            break

    all_shard_index_file = os.path.join(data_directory, f"shard_index.json")
    print(f"Shard index file is at {all_shard_index_file}")
    json.dump(shard_index, open(all_shard_index_file, 'w'))




def download_s2orc_full_text(download_commands, candidate_set, data_directory, data_type, string_match_map=None, reference_match_map=None, num_shards_to_use=None):
    for i, s2orc_shard_command in enumerate(download_commands):
        shard_paper_to_datasets_map = defaultdict(set)
        shard_index = defaultdict(list)

        output_path = f"s2orc_downloads/{s2orc_shard_command[2]}"
        data_url = eval(s2orc_shard_command[3])
        shard_id = get_shard_id_from_path(output_path, data_type=data_type)
        print(f"Starting processing of shard {shard_id}")
        start = time.perf_counter()
        if not os.path.exists(output_path):
            wget.download(data_url, out=output_path)
        # Time download
        end = time.perf_counter()
        print(f"\nTook {round(end - start, 3)} seconds to download shard {shard_id}.")
        start = end

        shard_file_name = os.path.join(data_directory, f"{shard_id}.jsonl.gz")
        shard_writer =  jsonlines.Writer(gzip.open(shard_file_name, 'wt'))

        # Load and match metadata rows
        shard = gzip.open(output_path, 'rt')
        s2orc_metadata = jsonlines.Reader(shard)
        searched_documents = 0
        hits = 0
        for doc in s2orc_metadata:
            paper_id = doc['paper_id']
            if paper_id in candidate_set:
                match = False
                if reference_match_map is not None and paper_id in reference_match_map:
                    for dataset_canonical_name in reference_match_map[paper_id]:
                        shard_paper_to_datasets_map[paper_id].add(("reference", dataset_canonical_name))
                    match = True
                if string_match_map is not None:
                    for section in doc["body_text"]:
                        section_text = section["text"].lower()
                        if len(section_text) > 0:
                            for dataset_variant, dataset_canonical_name in string_match_map.items():
                                if dataset_variant in section_text:
                                    shard_paper_to_datasets_map[paper_id].add(("mention", dataset_canonical_name))
                                    match=True
                                    break

                if match == True:
                    hits += 1
                    shard_writer.write(doc)
                    shard_index[shard_id].append(paper_id)
                searched_documents += 1

        end = time.perf_counter()
        shard_writer.close()
        print(f"Wrote {hits} hits from shard {shard_id} to {shard_file_name}.")
        print(f"Processing {searched_documents} docs in shard took {end - start} seconds.")

        paper_to_datasets_map_file = os.path.join(data_directory, f"paper_to_datasets_map_shard_{shard_id}.json")
        print(f"Paper-to-Datasets Map is at {paper_to_datasets_map_file}")
        shard_paper_to_datasets_map = {k: [list(tup) for tup in v] for k, v in shard_paper_to_datasets_map.items()}
        json.dump(shard_paper_to_datasets_map, open(paper_to_datasets_map_file, 'w'))

        all_shard_index_file = os.path.join(data_directory, f"shard_index_shard_{shard_id}.json")
        print(f"Shard index file is at {all_shard_index_file}")
        json.dump(shard_index, open(all_shard_index_file, 'w'))

        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"Deleted {output_path}")
        print("\n")
        if num_shards_to_use is not None and i + 1 >= num_shards_to_use:
            break

def accumulate_edges(citation_graph):
    citation_neighborhood_nodes = set()
    for (in_edges, out_edges) in citation_graph:
        for v in in_edges.values():
            citation_neighborhood_nodes.update(v)
        for v in out_edges.values():
            citation_neighborhood_nodes.update(v)
    return citation_neighborhood_nodes

def main():
    print("Starting.")
    print("Unpickling citation graphs.")
    full_data_download_commands = fetch_full_data_download_commands()
    metadata_download_commands = fetch_metadata_download_commands()
    citation_graph_edges = [os.path.join(caches_directory, f) for f in os.listdir(caches_directory) if f.startswith("citation_graph_radius_")]
    citation_graph_edges = [pickle.load(open(f, 'rb')) for f in citation_graph_edges]

    start = time.perf_counter()
    print(f"Loading paper IDs in neighborhood:")
    citation_neighborhood_nodes = accumulate_edges(citation_graph_edges)
    del citation_graph_edges

    print(f"Took {time.perf_counter() - start} seconds to load paper IDs in neighborhood.")

    # Write the object to disk and read from disk to free memory
    json.dump(list(citation_neighborhood_nodes), open("/tmp/nodes.json", 'w'))
    del citation_neighborhood_nodes
    citation_neighborhood_nodes = set(json.load(open("/tmp/nodes.json")))
    os.remove("/tmp/nodes.json")
    print(f"{len(citation_neighborhood_nodes)} documents in candidate set.")

    full_text_directory = os.path.join(caches_directory, "s2orc_full_texts")
    metadata_directory = os.path.join(caches_directory, "s2orc_metadata")
    for directory in [full_text_directory, metadata_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    reference_matches = json.load(open("/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/s2orc_papers_citing_datasets.json"))
    pwc_datasets_file = "/projects/ogma1/vijayv/dataset-recommendation/datasets.json"
    pwc_datasets = json.load(open(pwc_datasets_file))
    dataset_name_lookup_map = {}
    bad_names = set()
    var_length = 0
    for dataset_meta in pwc_datasets:
        dataset_name_lookup_map[dataset_meta["name"].lower()] = dataset_meta["name"]
        candidate_names = dataset_meta.get("variants", [])
        if dataset_meta.get("full_name", "") != "" and dataset_meta.get("full_name", "") != None:
            candidate_names.append(dataset_meta["full_name"])

        for candidate_name in list(set(candidate_names)):
            if candidate_name == dataset_meta["name"]:
                continue
            candidate_name = candidate_name.lower()
            if candidate_name in dataset_name_lookup_map:
                del dataset_name_lookup_map[candidate_name]
                bad_names.add(candidate_name)
            else:
                if candidate_name not in bad_names:
                    dataset_name_lookup_map[candidate_name] = dataset_meta["name"]

    download_s2orc_full_text(full_data_download_commands,
                        citation_neighborhood_nodes, full_text_directory,
                        data_type="full_text",
                        string_match_map=dataset_name_lookup_map,
                        reference_match_map=reference_matches)
    #download_s2orc_data(metadata_download_commands, citation_neighborhood_nodes, metadata_directory, data_type="metadata")

if __name__ == "__main__":
    main()