# Need to do:
# conda activate rapids-0.16
# In order to import cuml.

from collections import defaultdict, Counter
import json
import jsonlines
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
# can alternatively use the sklearn/CPU implementation instead:
# from sklearn.manifold import TSNE
# but it will take several hours (vs a couple minutes on GPU)
from cuml.manifold import TSNE
from join_scirex_and_s2orc import S2Metadata, S2OrcEntry, get_scirex_to_s2orc_mappings

def unpack_embeddings(path):
    graph = pickle.load(open(path, 'rb'))
    embeddings_matrix = graph['solver']['vertex_embeddings']
    name2id = graph['graph']['name2id']
    return embeddings_matrix, name2id

def compute_tsne(data, overwrite_cache=False):
    start = time.perf_counter()

    tsne_file = f"citation_graph_deepwalk_tsne.pkl"
    if os.path.exists(tsne_file) and not overwrite_cache:
        data_embedded =  pickle.load(open(tsne_file, 'rb'))
    else:
        visualizer = TSNE(perplexity=30, n_neighbors=90)
        data_embedded = visualizer.fit_transform(data)
        pickle.dump(data_embedded, open(tsne_file, 'wb'))
    end = time.perf_counter()
    print(f"Computing t-SNE took {end-start} seconds.")
    return data_embedded

def filter_outliers(arr, threshold=100):
    mean = np.mean(arr, axis=0)
    non_outlier_indices = []
    outlier_indices = []
    for i in range(len(arr)):
        if np.linalg.norm(arr[i] - mean) < threshold:
            non_outlier_indices.append(i)
        else:
            outlier_indices.append(i)
    return arr[non_outlier_indices], non_outlier_indices


def visualize_tsne(tsne_data, name2rowidx, scirex_node_id_mapping, sampling_rate = 0.01, initial_text_size=0.5, entity_type_to_label="Task"):
    scirex_idxs = np.array([name2rowidx[str(node_id)] for node_id in scirex_node_id_mapping.keys()])
    non_scirex_idxs = [d for d in range(len(tsne_data)) if d not in scirex_idxs]
    sampled_idxes = np.random.choice(non_scirex_idxs, int(sampling_rate * len(tsne_data)))

    plt.figure()
    task_topic_mapping = json.load(open("scirex_task_categorization.json"))
    doc_topics = []
    entity_counts = Counter()
    task_node_ids = defaultdict(list)
    entity_node_ids = defaultdict(list)
    for node_id, doc in scirex_node_id_mapping.items():
        doc_tasks = Counter()
        doc_entities = Counter()
        for relation in doc["n_ary_relations"]:
            if "Task" in relation and relation["Task"] not in doc_tasks:
                entity_counts[relation["Task"]] += 1
                doc_tasks[relation["Task"]] += 1
            if entity_type_to_label in relation and relation[entity_type_to_label] not in doc_entities:
                doc_entities[relation[entity_type_to_label]] += 1
        
        # Associate each paper with the most common labeled task in that paper.
        [(task_name, _)] = doc_tasks.most_common(1)
        doc_topics.append(task_topic_mapping[task_name])
        task_node_ids[task_name].append(node_id)

        [(entity_name, _)] = doc_entities.most_common(1)
        entity_node_ids[entity_name].append(node_id)

    doc_topics = np.array(doc_topics)
    scirex_data = tsne_data[scirex_idxs]

    scirex_data, non_outlier_indices = filter_outliers(scirex_data)
    doc_topics = doc_topics[non_outlier_indices]

    non_scirex_data_subsampled = tsne_data[sampled_idxes]
    non_scirex_data_subsampled, _ = filter_outliers(non_scirex_data_subsampled)
    print(f"scirex_data.shape: {scirex_data.shape}")
    print(f"non_scirex_data_subsampled.shape: {non_scirex_data_subsampled.shape}")
    palette = [
                'purple', # multimodal
                'blue', # vision
                'green', # games
                'yellow', # other
                'pink', # speech
                'brown', # ml_stat
                'red' #nlp
               ]
    ax2 = sns.scatterplot(x=non_scirex_data_subsampled[:, 0], y=non_scirex_data_subsampled[:, 1], marker=".", s=0.6, palette=['black'])
    ax2 = sns.scatterplot(x=scirex_data[:, 0], y=scirex_data[:, 1], marker="o", hue=doc_topics, s=2.0, palette=palette)
    # ax2 = sns.scatterplot(data=tsne_data_subsampled, marker='o', markersize=0.7)
    

        
    task_abbreviations = {
                        "Machine_Translation": "MT",
                        "Person_Pose_Estimation": "pose",
                        "Language_Modelling": "LM",
                        "Natural_Language_Inference": "NLI",
                        "Object_Detection": "detect",
                        "Image_Generation": "gan",
                        "Sentiment_Analysis": "sent",
                        "Speech_Recognition": "speech",
                        "Visual_Question_Answering": "vqa",
                        "Image_Classification": "img"
                    }

    for task_name in entity_counts:
        if task_name not in task_abbreviations:
            letters = []
            for tok in task_name.split("_"):
                if tok is "":
                    continue
                elif tok is "3D":
                    letters.append("3D")
                else:
                    letters.append(tok[0])
            shorthand = "".join(letters)
            task_abbreviations[task_name] = shorthand

    print(f"task_abbreviations: {json.dumps(task_abbreviations, sort_keys=True, indent=4)}")

    '''
    text_coordinates_to_draw = []
    for task_name, shorthand in task_abbreviations.items():
        for n in task_node_ids[task_name]:
            [c1, c2] =  tsne_data[name2rowidx[str(n)]]
            nearby_label_exists = False
            for i in range(len(text_coordinates_to_draw)):
                (task_label, coord, size) = text_coordinates_to_draw[i]
                if np.linalg.norm(np.array([c1, c2]) - coord) < 4 and task_label == task_name:
                    nearby_label_exists = True
                    break
            
            if not nearby_label_exists:
                text_coordinates_to_draw.append((task_name, np.array([c1, c2]), initial_text_size))
            else:
                text_coordinates_to_draw[i] = (task_name, coord, size + 0.25)
    '''

    text_coordinates_to_draw = []
    for entity_name, node_id in entity_node_ids.items():
        for n in entity_node_ids[entity_name]:
            [c1, c2] =  tsne_data[name2rowidx[str(n)]]
            nearby_label_exists = False
            for i in range(len(text_coordinates_to_draw)):
                (entity_label, coord, size) = text_coordinates_to_draw[i]
                if np.linalg.norm(np.array([c1, c2]) - coord) < 4 and entity_label == entity_name:
                    nearby_label_exists = True
                    break
            
            if not nearby_label_exists:
                text_coordinates_to_draw.append((entity_name, np.array([c1, c2]), initial_text_size))
            else:
                text_coordinates_to_draw[i] = (entity_name, coord, size + 0.25)

    for (task_name, coords, size) in text_coordinates_to_draw:
        c1 = coords[0]
        c2 = coords[1]
        ax2.text(c1, c2, task_name, color='black', fontsize=size, wrap=True)

    plt.legend(loc='lower right')
    plt.title(f"Citation graph embeddings annotated with {entity_type_to_label} labels")

    fig2 = ax2.get_figure()
    print(f"Saving plot...")
    savepath = f"/tmp/output_{entity_type_to_label.lower()}.png"
    fig2.savefig(savepath, dpi=800)
    print(f"Saved plot to {savepath}.")

def nearest_neighbors(embeddings, name2idx, input_s2orc_id, s2orc_id_to_node_id_mapping, nearest_neighbors=5):
    try:
        str(s2orc_id_to_node_id_mapping[input_s2orc_id])
    except:
        raise ValueError(f"s2orc id {input_s2orc_id} not in s2orc id-to-node id mapping")
    embedding_id = name2idx[str(s2orc_id_to_node_id_mapping[input_s2orc_id])]
    idx_to_node_id = {v:k for (k,v) in name2idx.items()}
    node_id_to_s2orc_id = {v:k for (k,v) in s2orc_id_to_node_id_mapping.items()}
    vector = embeddings[embedding_id]
    start = time.perf_counter()
    distances = [np.linalg.norm(vector - v) for v in embeddings]
    end = time.perf_counter()
    sorted_args = np.argsort(distances)[1:1+nearest_neighbors]
    print(f"Took {end-start} seconds to compute nearest neighbors over {len(embeddings)} embeddings")
    for array_idx in sorted_args:
        node_id = idx_to_node_id[array_idx]
        print(f"{distances[array_idx]}\t\t{node_id_to_s2orc_id[int(node_id)]}")

def main():
    scirex_train = jsonlines.open("/projects/ogma1/vijayv/SciREX/scirex_dataset/release_data/train.jsonl")
    scirex_test = jsonlines.open("/projects/ogma1/vijayv/SciREX/scirex_dataset/release_data/test.jsonl")
    scirex_dev = jsonlines.open("/projects/ogma1/vijayv/SciREX/scirex_dataset/release_data/dev.jsonl")
    all_scirex_documents = []
    for split in [scirex_train, scirex_test, scirex_dev]:
        for doc in split:
            all_scirex_documents.append(doc)


    embeddings, name2id = unpack_embeddings("/projects/ogma1/vijayv/graphvite/graph.pkl")

    tsne_data = compute_tsne(embeddings, overwrite_cache=False)
    scirex_to_s2orc_mappings = get_scirex_to_s2orc_mappings()
    s2orc_id_to_node_id_mapping = json.load(open("/projects/ogma1/vijayv/ScigraphIE/node_id_mapping.json"))

    node_id_to_scirex_metadata = {}
    miss_counter1 = 0
    miss_counter2 = 0
    for doc in all_scirex_documents:
        scirex_docid = doc['doc_id']
        if scirex_docid not in scirex_to_s2orc_mappings:
            miss_counter1 += 1
            continue
        if scirex_to_s2orc_mappings[scirex_docid] not in s2orc_id_to_node_id_mapping:
            miss_counter2 += 1
            continue
        node_id = s2orc_id_to_node_id_mapping[scirex_to_s2orc_mappings[scirex_docid]]
        node_id_to_scirex_metadata[node_id] = doc

    visualize_tsne(tsne_data, name2id, node_id_to_scirex_metadata, entity_type_to_label="Task")
    visualize_tsne(tsne_data, name2id, node_id_to_scirex_metadata, entity_type_to_label="Method")
    visualize_tsne(tsne_data, name2id, node_id_to_scirex_metadata, entity_type_to_label="Metric")
    visualize_tsne(tsne_data, name2id, node_id_to_scirex_metadata, entity_type_to_label="Material")

    input_s2orc_id = "630188"
    nearest_neighbors(embeddings, name2id, input_s2orc_id, s2orc_id_to_node_id_mapping)

if __name__ == "__main__":
    main()