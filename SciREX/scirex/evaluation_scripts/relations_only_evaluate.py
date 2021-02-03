import argparse
from itertools import combinations
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve

from scirex.metrics.clustering_metrics import match_predicted_clusters_to_gold
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex_utilities.entity_utils import used_entities
from scirex_utilities.json_utilities import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--ner-file")
parser.add_argument("--clusters-file")
parser.add_argument("--relations-file")
parser.add_argument("--dev-gold-file")
parser.add_argument("--dev-ner-file")
parser.add_argument("--dev-clusters-file")
parser.add_argument("--dev-relations-file")
parser.add_argument("--choose-dev-thresholds", action='store_true')
parser.add_argument("--choose-with-retrieval-metrics", action='store_true', help="If unset, then we will choose the best threshold with retrieval metrics instead of classification metrics.")
parser.add_argument("--choose-with-2-ary", action='store_true', help="If unset, then we will choose the best threshold with 2-ary relation metrics instead of 4-ary.")
parser.add_argument("--file-suffix", default="scratch", type=str)

def construct_valid_thresholds():
    return np.arange(0, 1, 0.001)

def has_all_mentions(doc, relation):
    has_mentions = all(len(doc["clusters"][x[1]]) > 0 for x in relation)
    return has_mentions


def convert_to_dict(data):
    return {x["doc_id"]: x for x in data}


def ner_metrics(gold_data, predicted_data):
    mapping = {}
    for doc in gold_data:
        predicted_doc = predicted_data[doc["doc_id"]]
        predicted_spans = predicted_doc["ner"]
        gold_spans = doc["ner"]

        mapping[doc["doc_id"]] = map_predicted_spans_to_gold(predicted_spans, gold_spans)

    return mapping


def match_predicted_clusters_with_gold(gold_data, predicted_clusters, span_map):
    mappings = {}
    for doc in gold_data:
        predicted_doc = predicted_clusters[doc["doc_id"]]
        _, mapping = match_predicted_clusters_to_gold(
            predicted_doc["clusters"], doc["coref"], span_map[doc["doc_id"]], doc['words']
        )
        mappings[doc["doc_id"]] = mapping
    return mappings


def get_types_of_clusters(predicted_ner, predicted_clusters):
    for doc_id in predicted_clusters:
        clusters = predicted_clusters[doc_id]["clusters"]
        ner = {(x[0], x[1]): x[2] for x in predicted_ner[doc_id]["ner"]}

        predicted_clusters[doc_id]["types"] = {}
        for c, spans in clusters.items():
            types = set([ner[tuple(span)] for span in spans])
            if len(types) == 0:
                predicted_clusters[doc_id]["types"][c] = "Empty"
                continue
            predicted_clusters[doc_id]["types"][c] = list(types)[0]


def compute_weighted_auc(gold_data,
                         predicted_ner,
                         predicted_relations,
                         predicted_cluster_to_gold_cluster_map,
                         best_threshold,
                         n=4):
    pred_scores = []
    y_labels = []
    y_preds = []
    y_preds_fixed = []
    for types in combinations(used_entities, n):
        for doc in gold_data:
            predicted_data = predicted_relations[doc["doc_id"]]
            mapping = predicted_cluster_to_gold_cluster_map[doc["doc_id"]]

            gold_relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]
            gold_relations = set([x for x in gold_relations if has_all_mentions(doc, x)])

            seen_relations = set()
            for relation_tuple in predicted_data["predicted_relations"]:
                relation = relation_tuple[0]
                relation_score = relation_tuple[1]
                relation_pred = relation_tuple[2]
                ordered_relation = tuple([mapping.get(v, v) for v in relation])
                ordered_relation = dict(zip(used_entities, ordered_relation))
                ordered_relation = tuple((t, ordered_relation[t]) for t in types)
                if ordered_relation not in seen_relations and len(gold_relations) > 0:
                    seen_relations.add(ordered_relation)
                    pred_scores.append(relation_score)
                    y_labels.append(ordered_relation in gold_relations)
                    y_preds.append(relation_score >= best_threshold)
                    y_preds_fixed.append(relation_pred)

            for relation_tuple in gold_relations.difference(seen_relations):
                pred_scores.append(0.0)
                y_labels.append(True)
                # y_preds.append(predicted_score >= thresh)
                y_preds.append(False)
                y_preds_fixed.append(False)


    average_precision = average_precision_score(y_labels, pred_scores, average='macro')

    f1 = f1_score(y_labels, y_preds, average='macro')
    classification_precision = precision_score(y_labels, y_preds, average='macro')
    classification_recall = recall_score(y_labels, y_preds, average='macro')
    print(f"Best threshold values:\np: {classification_precision}\tr: {classification_recall}\tf1: {f1}")

    classification_precision_fixed = precision_score(y_labels, y_preds_fixed, average='macro')
    classification_recall_fixed = recall_score(y_labels, y_preds_fixed, average='macro')
    f1_fixed = f1_score(y_labels, y_preds_fixed, average='macro')
    print(f"Fixed threshold values:\np: {classification_precision_fixed}\tr: {classification_recall_fixed}\tf1: {f1_fixed}")

    precision, recall, thresholds = precision_recall_curve(y_labels, pred_scores)
    
    pr_at_thresholds = {}

    best_approximate_threshold = -10000
    for t, p, r in zip(thresholds, precision, recall):
        prf1 = {}
        prf1["precision"] = p
        prf1["recall"] = r
        prf1["f1"] = f1
        pr_at_thresholds[t] = prf1
        if np.abs(t - best_threshold) < np.abs(t - best_approximate_threshold):
            best_approximate_threshold = t

    return average_precision, f1, f1_fixed, pr_at_thresholds, best_approximate_threshold

def compute_cluster_width(doc, relation):
    min_mention = int(1e100)
    max_mention = -1
    for (_, entity_name) in relation:
        if len(doc["coref"][entity_name]) == 0:
            continue
        min_mention = min(min_mention, min([span[0] for span in doc["coref"][entity_name]]))
        max_mention = max(max_mention, max([span[1] for span in doc["coref"][entity_name]]))
    return float(max_mention - min_mention) / len(doc["words"])


def compute_average_entity_cluster_width(doc, relation):
    all_widths = []
    for (_, entity_name) in relation:
        if len(doc["coref"][entity_name]) == 0:
            continue
        min_mention = min([span[0] for span in doc["coref"][entity_name]])
        max_mention = max([span[1] for span in doc["coref"][entity_name]])
        assert max_mention >= min_mention
        all_widths.append(max_mention - min_mention)
    return np.mean(all_widths) / len(doc["words"])



def compute_average_minimum_cluster_distance(doc, relation):
    cluster_pair_distances = []
    pairs_already_checked = set()
    for i, (_, entity_name_a) in enumerate(relation):
        for j, (_, entity_name_b) in enumerate(relation):
            if tuple(sorted([i,j])) in pairs_already_checked or i == j:
                continue
            pairs_already_checked.add(tuple(sorted([i,j])))
            mention_a_starts = [span[0] for span in doc["coref"][entity_name_a]]
            mention_a_ends = [span[1] for span in doc["coref"][entity_name_a]]
            mention_b_starts = [span[0] for span in doc["coref"][entity_name_b]]
            mention_b_ends = [span[0] for span in doc["coref"][entity_name_b]]
            distances = []
            for b_idx in mention_b_starts:
                for a_idx in mention_a_ends:
                    if b_idx > a_idx:
                        distances.append(b_idx - a_idx)
            for a_idx in mention_a_starts:
                for b_idx in mention_b_ends:
                    if a_idx > b_idx:
                        distances.append(a_idx - b_idx)
            if len(distances) == 0:
                continue
            minimum_cluster_distance = np.mean(distances)
            cluster_pair_distances.append(minimum_cluster_distance)

    return np.mean(cluster_pair_distances) / len(doc["words"])

def compute_relations_metrics(gold_data, predicted_ner, predicted_salient_clusters, predicted_relations, predicted_cluster_to_gold_cluster_map, thresh=None, n=4, cluster_width_bucket=None):
    retrieval_metrics = []
    num_predicted = 0
    num_labeled = 0
    num_matched = 0

    sum_average_precision = 0.0
    number_of_documents = 0

    y_labels = []
    y_preds = []

    for types in combinations(used_entities, n):
        for doc in gold_data:
            predicted_data = predicted_relations[doc["doc_id"]]
            mapping = predicted_cluster_to_gold_cluster_map[doc["doc_id"]]

            if thresh is None:
                relations = list(set([
                    tuple([mapping.get(v, v) for v in x[0]])
                    for x in predicted_data["predicted_relations"]
                    if x[2] == 1
                ]))
            else:
                relations = list(set([
                    tuple([mapping.get(v, v) for v in x[0]])
                    for x in predicted_data["predicted_relations"]
                    if x[1] >= thresh
                ]))

            relations = [dict(zip(used_entities, x)) for x in relations]
            relations = set([tuple((t, x[t]) for t in types) for x in relations])

            gold_relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]
            gold_relations = set([x for x in gold_relations if has_all_mentions(doc, x)])

            if cluster_width_bucket is not None:
                gold_relations_in_bucket = []
                for relation in list(gold_relations):
                    if compute_average_minimum_cluster_distance(doc, relation) > cluster_width_bucket[0] and compute_average_minimum_cluster_distance(doc, relation) <= cluster_width_bucket[1]:
                        gold_relations_in_bucket.append(relation)
                gold_relations = set(gold_relations_in_bucket)

            relations_seen = set()
            relations_with_scores = []
            for relation_tuple in predicted_data["predicted_relations"]:
                relation_remapped = tuple([mapping.get(v, v) for v in relation_tuple[0]])
                relation_remapped = dict(zip(used_entities, relation_remapped))
                relation_remapped = tuple((t, relation_remapped[t]) for t in types)

                relation_score = relation_tuple[1]
                relation_pred = relation_tuple[2]
                if relation_remapped in relations_seen:
                    continue
                relations_seen.add(relation_remapped)
                relations_with_scores.append((relation_remapped, relation_score, relation_pred))

            relations_sorted = sorted(relations_with_scores, key=lambda x: x[1], reverse=True)
            relations_sorted = [x[0] for x in relations_sorted]

            y_preds_doc  = []
            y_labels_doc = []
            for relation_tuple in relations_with_scores:
                pred = relation_tuple[1] >= thresh if thresh is not None else relation_tuple[2] == 1
                label = relation_tuple[0] in gold_relations
                y_preds_doc.append(pred)
                y_labels_doc.append(label)


            if len(gold_relations) > 0:
                average_precision = 0.0
                prev_recall = 0.0
                for k in range(1, len(relations_sorted) + 1):
                    relations_up_to_k = set(relations_sorted[:k])
                    matched_up_to_k = relations_up_to_k & gold_relations
                    precision = len(matched_up_to_k) / len(relations_up_to_k)
                    recall = len(matched_up_to_k) / len(gold_relations)
                    assert recall >= prev_recall

                    average_precision += precision * (recall - prev_recall)
                    prev_recall = recall

            try:
                matched = relations & gold_relations
            except:
                breakpoint()

            metrics = {
                "p": len(matched) / (len(relations) + 1e-7),
                "r": len(matched) / (len(gold_relations) + 1e-7),
            }
            metrics["f1"] = 2 * metrics["p"] * metrics["r"] / (metrics["p"] + metrics["r"] + 1e-7)


            if len(gold_relations) > 0:
                retrieval_metrics.append(metrics)

                num_predicted += len(relations)
                num_labeled += len(gold_relations)
                num_matched += len(matched)

                sum_average_precision += average_precision
                number_of_documents += 1

                # The above predictions and labels were only covering relations that we did predict. We also need to count
                # relations that were labeled true, but we did not predict (equivalently, we predicted with score 0.0).
                for relation_tuple in gold_relations.difference(relations_seen):
                    y_labels_doc.append(True)
                    y_preds_doc.append(False)

                y_labels.extend(y_labels_doc)
                y_preds.extend(y_preds_doc)

    retrieval_metrics_df = pd.DataFrame(retrieval_metrics)
    mean_average_precision = sum_average_precision / number_of_documents
    return retrieval_metrics_df, mean_average_precision, y_labels, y_preds, num_predicted, num_labeled, num_matched


def summarize_relation_extraction_metrics(retrieval_metrics_df, classification_y_labels, classification_y_preds, average='binary'):
    retrieval_metric_summary = retrieval_metrics_df.describe().loc['mean'][['p', 'r', 'f1']]
    f1 = f1_score(classification_y_labels, classification_y_preds, average=average)
    classification_precision = precision_score(classification_y_labels, classification_y_preds, average=average)
    classification_recall = recall_score(classification_y_labels, classification_y_preds, average=average)
    classification_metrics = {
                                "f1": f1,
                                "p": classification_precision,
                                "r": classification_recall
                            }
    return retrieval_metric_summary, classification_metrics


def prepare_data(gold_file, ner_file, clusters_file, relations_file):
    gold_data = load_jsonl(gold_file)
    for d in gold_data:
        merge_method_subrelations(d)
        d["clusters"] = d["coref"]

    predicted_ner = convert_to_dict(load_jsonl(ner_file))
    predicted_salient_clusters = convert_to_dict(load_jsonl(clusters_file))
    for d, doc in predicted_salient_clusters.items() :
        if 'clusters' not in doc :
            merge_method_subrelations(doc)
            doc['clusters'] = {x:v for x, v in doc['coref'].items() if len(v) > 0}

    predicted_relations = convert_to_dict(load_jsonl(relations_file))

    predicted_span_to_gold_span_map: Dict[str, Dict[tuple, tuple]] = ner_metrics(gold_data, predicted_ner)
    get_types_of_clusters(predicted_ner, predicted_salient_clusters)
    get_types_of_clusters(convert_to_dict(gold_data), convert_to_dict(gold_data))
    predicted_cluster_to_gold_cluster_map = match_predicted_clusters_with_gold(
        gold_data, predicted_salient_clusters, predicted_span_to_gold_span_map
    )
    return gold_data, predicted_ner, predicted_salient_clusters, predicted_relations, predicted_cluster_to_gold_cluster_map


def draw_pr_curve(results_dict, average_precision, best_thresh, file_suffix="scratch"):
    thresholds = list(results_dict.keys())
    recalls = [x['recall'] for x in results_dict.values()]
    precisions = [x['precision'] for x in results_dict.values()]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lines = ax.plot(recalls, precisions, lw=2, color='green')
    lines = ax.plot([results_dict[best_thresh]['recall']], [results_dict[best_thresh]['precision']], marker='o', color='green', markersize=12, label="Best Operating Point (from dev set)")
    # lines = ax.plot(recalls, thresholds, color='blue', label="Threshold/Recall Curve")

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision/Threshold/Probability')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.grid()
    ax.legend()

    rounded_ap = round(average_precision, 4)
    rounded_best_p = round(results_dict[best_thresh]['f1'], 4)
    rounded_thresh = round(best_thresh, 3)
    ax.set_title(f'Precision-Recall ({file_suffix}) - Macro-AP={rounded_ap}\nAt marked threshold {rounded_thresh}, Macro-F1 is {rounded_best_p}')
    fname = f"/tmp/pr_curve_{file_suffix}.png"
    fig.savefig(fname)
    print(f"Wrote PR curve to {fname}")



def draw_pr_curve_against_threshold(results_dict, average_precision, best_thresh, file_suffix="scratch"):
    thresholds = list(results_dict.keys())
    recalls = [x['recall'] for x in results_dict.values()]
    precisions = [x['precision'] for x in results_dict.values()]
    positive_sizes = [x['num_positive'] for x in results_dict.values()]
    total_positive = positive_sizes[0]
    positive_sizes = [float(pos) / total_positive for pos in positive_sizes]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    lines = ax.plot(thresholds, recalls, lw=2, color='orange', label="Threshold/Recall Curve")
    lines = ax.plot(thresholds, precisions, lw=2, color='blue', label="Threshold/Precision Curve")
    lines = ax.plot([best_thresh], [results_dict[best_thresh]['precision']], marker='o', color='green', markersize=12)
    lines = ax.plot([best_thresh], [results_dict[best_thresh]['recall']], marker='o', color='green', markersize=12)
    lines = ax.plot(thresholds, positive_sizes, lw=2, color='red', label="Threshold/Fraction of predictions above threshold")

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision/Recall/Fraction Positive')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.grid()
    ax.legend(loc='best')

    rounded_ap = round(average_precision, 4)
    rounded_best_p = round(results_dict[best_thresh]['f1'], 4)
    rounded_thresh = round(best_thresh, 3)
    ax.set_title(f'Precision-Recall ({file_suffix}) - Macro-AP={rounded_ap}\nAt marked threshold {rounded_thresh}, Macro-F1 is {rounded_best_p}')
    fname = f"/tmp/pr_curve_{file_suffix}_w_thresholds.png"
    fig.savefig(fname)
    print(f"Wrote PR curve to {fname}")

def draw_histogram(data, num_bins = 200, file_suffix="scratch"):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # matplotlib histogram
    ax.hist(data, color = 'blue', edgecolor = 'black', bins = num_bins)

    # Add labels
    ax.set_title('Histogram of probabilities')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Counts')
    fname = f"/tmp/probability_histogram_{file_suffix}.png"
    fig.savefig(fname)
    print(f"Wrote probability histogram to {fname}")

def main(args):
    processed_data = prepare_data(args.gold_file, args.ner_file, args.clusters_file, args.relations_file)
    gold_data, predicted_ner, predicted_salient_clusters, predicted_relations, predicted_cluster_to_gold_cluster_map = processed_data


    all_probabilities = []
    all_predictions = []
    all_positive_probabilities = []
    for doc in gold_data:
        predicted_data = predicted_relations[doc["doc_id"]]
        probabilities = [x[1] for x in predicted_data["predicted_relations"]]
        predictions = [x[2] for x in predicted_data["predicted_relations"]]
        positive_probabilities = [x[1] for x in zip(predictions, probabilities) if x[0] == 1]

        all_probabilities.extend(probabilities)
        all_predictions.extend(predictions)
        all_positive_probabilities.extend(positive_probabilities)

    draw_histogram(all_probabilities, file_suffix=args.file_suffix)


    if args.choose_dev_thresholds:
        dev_processed_data = prepare_data(args.dev_gold_file, args.dev_ner_file, args.dev_clusters_file, args.dev_relations_file)
        dev_gold_data, dev_predicted_ner, dev_predicted_salient_clusters, dev_predicted_relations, dev_predicted_cluster_to_gold_cluster_map = dev_processed_data

        best_threshold = -1
        best_f1 = -1
        n = 2 if args.choose_with_2_ary else 4
        threshold_values = []
        for candidate_thresh in tqdm(construct_valid_thresholds()):
            retrieval_metrics_df, _, y_labels, y_preds, _, _, _, _ = compute_relations_metrics(
                                                    dev_gold_data,
                                                    dev_predicted_ner,
                                                    dev_predicted_salient_clusters,
                                                    dev_predicted_relations,
                                                    dev_predicted_cluster_to_gold_cluster_map,
                                                    thresh=candidate_thresh,
                                                    n=n)
            retrieval_metrics, classification_metrics = summarize_relation_extraction_metrics(retrieval_metrics_df,
                                                                                              y_labels,
                                                                                              y_preds,
                                                                                              average='binary')
            if retrieval_metrics is None and classification_metrics is None:
                continue
            f1 = retrieval_metrics['f1'] if args.choose_with_retrieval_metrics else classification_metrics['f1']
            prf1 = dict(zip(["f1", "precision", "recall"], [retrieval_metrics['f1'], retrieval_metrics['p'], retrieval_metrics['r']]))
            threshold_values.append((candidate_thresh, prf1))
            if f1 > best_f1:
                 best_f1 = f1
                 best_threshold = candidate_thresh

        thresh = best_threshold

        metric_objective_type = "retrieval metric" if args.choose_with_retrieval_metrics else "classification metric"
        print(f"Best threshold is {round(thresh, 4)}, with dev-set {metric_objective_type} (n={n}) value of {round(best_f1, 4)}")
    else:
        # Use pre-thresholded predictions
        thresh = None

    for n in [2, 4]:
        thresh_string = str(thresh) if thresh is not None else "<fixed>"
        print(f"At threshold {thresh_string}:")
        retrieval_metrics_df, mean_average_precision, y_labels, y_preds, _, _, _ = compute_relations_metrics(gold_data,
                                                predicted_ner,
                                                predicted_salient_clusters,
                                                predicted_relations,
                                                predicted_cluster_to_gold_cluster_map,
                                                n=n,
                                                thresh=thresh)

        retrieval_metrics, classification_metrics = summarize_relation_extraction_metrics(retrieval_metrics_df,
                                                                                            y_labels,
                                                                                            y_preds,
                                                                                            average='macro')
        print(f"Relation Metrics n={n}")
        print(retrieval_metrics)
        print(f"Retrieval MAP (mean average precision): {mean_average_precision}")

        classification_precision = classification_metrics['p']
        classification_recall = classification_metrics['r']
        f1 = classification_metrics['f1']
        print(f"Classification Precision: {classification_precision}")
        print(f"Classification Recall: {classification_recall}")
        print(f"Classification F1: {f1}\n")

    """
        print("\nComputing AUC using sklearn built-in functions")
        if not args.choose_dev_thresholds:
            best_threshold = 0.5
        average_precision, f1, f1_fixed, pr_at_thresholds, closest_thresh_to_best = compute_weighted_auc(
                                        gold_data,
                                        predicted_ner,
                                        predicted_relations,
                                        predicted_cluster_to_gold_cluster_map,
                                        best_threshold,
                                        n=n)
        try:
            draw_pr_curve(pr_at_thresholds, average_precision, closest_thresh_to_best, file_suffix=f"{args.file_suffix}_s")
        except:
            breakpoint()

        print(f"Average precision across all thresholds (sklearn): {average_precision}")
        print(f"Classification F1 at model-chosen threshold (sklearn): {f1_fixed}\n\n")

        classification_precisions = []
        retrieval_precisions = []

        threshold_values = {}
        datasize_values = {}
        for candidate_thresh in tqdm(construct_valid_thresholds()):
            retrieval_metrics_df, _, y_labels, y_preds, num_predicted, num_labeled, num_matched = compute_relations_metrics(gold_data,
                                        predicted_ner,
                                        predicted_salient_clusters,
                                        predicted_relations,
                                        predicted_cluster_to_gold_cluster_map,
                                        n=n,
                                        thresh=candidate_thresh)
            retrieval_metrics, classification_metrics = summarize_relation_extraction_metrics(retrieval_metrics_df,
                                                                                                y_labels,
                                                                                                y_preds,
                                                                                                average='macro')
            if retrieval_metrics is None and classification_metrics is None:
                continue
            f1 = retrieval_metrics['f1'] if args.choose_with_retrieval_metrics else classification_metrics['f1']

            prf1 = {}
            prf1["f1"] = classification_metrics['f1']
            prf1["precision"] = classification_metrics['p']
            prf1["recall"] = classification_metrics['r']
            prf1["num_positive"] = num_predicted


            threshold_values[candidate_thresh] = prf1
            datasize_values[candidate_thresh] = (num_predicted, num_labeled, num_matched)

            retrieval_precisions.append(retrieval_metrics['p'])
            classification_precisions.append(classification_metrics['p'])

        draw_pr_curve(threshold_values, average_precision, best_threshold, file_suffix=f"{args.file_suffix}_n_{n}")
        draw_pr_curve_against_threshold(threshold_values, average_precision, best_threshold, file_suffix=f"{args.file_suffix}_n_{n}")
    """

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
