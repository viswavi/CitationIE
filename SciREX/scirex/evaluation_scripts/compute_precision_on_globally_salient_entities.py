import argparse
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict

import pandas as pd

from scirex.evaluation_scripts.salient_bootstrap_comparison_bucketing import draw_box_plot_with_error_bars
from scirex.metrics.paired_bootstrap import eval_with_hierarchical_paired_bootstrap
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex_utilities.entity_utils import used_entities
from scirex_utilities.json_utilities import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--train-gold-file")
parser.add_argument("--test-gold-file")
parser.add_argument("--salient-mentions-files-a", help="Salient mentions from system A", nargs='+', type=str)
parser.add_argument("--salient-mentions-files-b", help="Salient mentions from system B", nargs='+', type=str)
parser.add_argument("--num-buckets", default=4, type=int)
parser.add_argument("--use-local-coref-clusters", action="store_true")


def convert_to_dict(data):
    return {x["doc_id"]: x for x in data}

def look_up_in_coreferences(span_tuple, coreferences):
    for i, coref in enumerate(coreferences):
        if span_tuple in coref:
            return i
    raise ValueError("Should not be reachable")

def standardize_span(words):
    span_surface_form = [w.lower() for w in words]
    return tuple(span_surface_form)

def build_global_coreference_lists(gold_data):
    coreferences = []
    for doc in tqdm(gold_data):
        for cluster in list(doc["coref"].values()) + list(doc["coref_non_salient"].values()):
            coreferring_cluster_idx = -1
            cluster_spans = []
            for span in cluster:
                span_surface_form = [w.lower() for w in doc["words"][span[0]:span[1]]]
                standardized_surface_form = tuple(span_surface_form)
                cluster_spans.append(standardized_surface_form)
                for i, existing_coref_cluster in enumerate(coreferences):
                    if coreferring_cluster_idx is not -1:
                        break
                    if standardized_surface_form in existing_coref_cluster:
                        coreferring_cluster_idx = i
                        break
            
            if coreferring_cluster_idx == -1:
                coreferences.append([])
                coreferring_cluster_idx = len(coreferences)-1
            coreferences[coreferring_cluster_idx].extend(cluster_spans)
            coreferences[coreferring_cluster_idx] = list(set(coreferences[coreferring_cluster_idx]))
    return coreferences          

def compute_global_saliency_dict_global_coreferences(gold_data, global_coref_clusters, counts_threshold=25):
    # Each value contains a pair of counts: [number_salient, number_non_salient]
    saliency_counts = [[0, 0] for _ in global_coref_clusters]
    saliency_dict = defaultdict(lambda: [0, 0])
    for doc in tqdm(gold_data):
        for cluster in doc["coref"].values():
            for span in cluster:
                standardized_surface_form = standardize_span(doc["words"][span[0]:span[1]])
                coref_cluster_idx = look_up_in_coreferences(standardized_surface_form, global_coref_clusters)
                saliency_counts[coref_cluster_idx][0] += 1
        for cluster in doc["coref_non_salient"].values():
            cluster_set = set()
            for span in cluster:
                standardized_surface_form = standardize_span(doc["words"][span[0]:span[1]])
                coref_cluster_idx = look_up_in_coreferences(standardized_surface_form, global_coref_clusters)
                saliency_counts[coref_cluster_idx][1] += 1

    saliency_ratios = {}
    discarded_entities = 0
    zerolens = [c for c in saliency_counts if sum(c) == 0]
    for idx, coref_cluster in enumerate(global_coref_clusters):
        counts = saliency_counts[idx]
        saliency_count = counts[0]
        total_entity_count = sum(counts)
        if total_entity_count == 0 or total_entity_count < counts_threshold:
            discarded_entities += 1
            continue
        saliency_rate = float(saliency_count)/total_entity_count
        for surface_form in coref_cluster:
            saliency_ratios[surface_form] = saliency_rate
    print(f"Discarded {discarded_entities} entities which appeared less than {counts_threshold} times")
    return saliency_ratios

def compute_global_saliency_dict_local_coreferences(gold_data, counts_threshold=25):
    # Each value contains a pair of counts: [number_salient, number_non_salient]
    saliency_dict = defaultdict(lambda: [0, 0])
    for doc in tqdm(gold_data):
        for cluster in doc["coref"].values():
            cluster_set = set()
            for span in cluster:
                standardized_surface_form = standardize_span(doc["words"][span[0]:span[1]])
                if standardized_surface_form in cluster_set:
                    continue
                cluster_set.add(standardized_surface_form)
                saliency_dict[standardized_surface_form][0] += len(cluster)
        for cluster in doc["coref_non_salient"].values():
            cluster_set = set()
            for span in cluster:
                standardized_surface_form = standardize_span(doc["words"][span[0]:span[1]])
                if standardized_surface_form in cluster_set:
                    continue
                cluster_set.add(standardized_surface_form)
                saliency_dict[standardized_surface_form][1] += len(cluster)

    saliency_ratios = {}
    discarded_entities = 0
    for surface_form, counts in saliency_dict.items():
        saliency_count = counts[0]
        total_entity_count = sum(counts)
        if total_entity_count < counts_threshold:
            discarded_entities += 1
            continue
        saliency_rate = float(saliency_count)/total_entity_count
        saliency_ratios[surface_form] = saliency_rate
    return saliency_ratios

def salent_mentions_metrics(gold_data, predicted_salient_mentions_a_list, predicted_salient_mentions_b_list, saliency_rate_dict, saliency_rate_buckets, eval_type="fpr"):
    assert eval_type in ["fpr", "fnr", "f1", "macro_f1"]
    all_metrics = []
    predicted = 0
    gold = 0
    matched = 0

    bucketed_eval_comparison = {}

    system_predictions = [predicted_salient_mentions_a_list, predicted_salient_mentions_b_list]
    for bucket in saliency_rate_buckets:
        bucket_formatted = str((round(bucket[0], 3), round(bucket[1], 3)))
        print(f"Bucket: {bucket_formatted}")
        bucket_values_a = [[] for _ in predicted_salient_mentions_a_list]
        bucket_values_b = [[] for _ in predicted_salient_mentions_b_list]
        bucket_values = [bucket_values_a, bucket_values_b]

        pred_len = None
        for doc in gold_data:
            gold_salient_spans = [span for coref_cluster in doc['coref'].values() for span in coref_cluster]

        y_labels = None
        for i in range(2):
            # For prediction in A, B:
            for j, mention_predictions in enumerate(system_predictions[i]):
                y_labels_single_run = []
                for doc in gold_data:
                    gold_salient_spans = [span for coref_cluster in doc['coref'].values() for span in coref_cluster]
                    predicted_doc = mention_predictions[doc["doc_id"]]
                    for [start_span, end_span, saliency, _] in predicted_doc["saliency"]:
                        standardized_span_form = standardize_span(doc["words"][start_span:end_span])
                        if standardized_span_form not in saliency_rate_dict:
                            continue
                        saliency_rate = saliency_rate_dict[standardized_span_form]
                        if not (saliency_rate >= bucket[0] and saliency_rate < bucket[1]):
                            # Only count entities with global saliency rate in bucket.
                            continue
                        if eval_type == "fpr":
                            # Compute FPR
                            if (start_span, end_span) not in gold_salient_spans:
                                # Only count labeled negatives
                                bucket_values[i][j].append(1.0 if saliency else 0.0)
                        elif eval_type == "fnr":
                            # Compute FNR
                            if (start_span, end_span) in gold_salient_spans:
                                # Only count labeled positives
                                bucket_values[i][j].append(0.0 if saliency else 1.0)
                        else:
                            # Compute F1
                            true_label = (start_span, end_span) in gold_salient_spans
                            bucket_values[i][j].append(saliency)
                            y_labels_single_run.append(true_label)
                if y_labels is None:
                    y_labels = y_labels_single_run
                else:
                    assert y_labels == y_labels_single_run


        for bucket_values_list in bucket_values:
            for system_values in bucket_values_list:
                if pred_len is None:
                    pred_len = len(system_values)
                else:
                    assert pred_len == len(system_values)
        print(f"Samples in bucket: {pred_len}")

        sys1_values_list = bucket_values[0]
        sys2_values_list = bucket_values[1]
        # The bootstrap script expects a list of gold values, but here the "system" values are already 
        # comparisons with gold, so just pass in a list of Nones to satisfy the input.
        if eval_type in ["fpr", "fnr"]:
            gold_mentions = [None for _ in sys1_values_list[0]]
            sys1_summary, sys2_summary, p_value_lose, p_value_win = eval_with_hierarchical_paired_bootstrap(gold_mentions, sys1_values_list, sys2_values_list,
                                    num_samples=1000, sample_ratio=0.5,
                                    eval_type="avg", return_results=True)
        else:
            try:
                gold_mentions = y_labels
                f1_type = "f1" if eval_type == "f1" else "macro-f1"
                sys1_summary, sys2_summary, p_value_lose, p_value_win = eval_with_hierarchical_paired_bootstrap(gold_mentions, sys1_values_list, sys2_values_list,
                                        num_samples=1000, sample_ratio=0.5,
                                        eval_type=f1_type, return_results=True)
            except:
                breakpoint()
        bucketed_eval_comparison[bucket_formatted] = {"base": [list(sys1_summary), p_value_lose], "diff": [list(sys2_summary), p_value_win]}
    return bucketed_eval_comparison
        

def main(args):
    train_gold_data = load_jsonl(args.train_gold_file)
    for d in train_gold_data:
        merge_method_subrelations(d)
        d["clusters"] = d["coref"]

    test_gold_data = load_jsonl(args.test_gold_file)
    for d in test_gold_data:
        merge_method_subrelations(d)
        d["clusters"] = d["coref"]
    
    if args.use_local_coref_clusters:
        saliency_dict = compute_global_saliency_dict_local_coreferences(train_gold_data, counts_threshold=2)
        coref_string = "local_coref"
    else:
        global_coref_clusters = build_global_coreference_lists(train_gold_data)
        saliency_dict = compute_global_saliency_dict_global_coreferences(train_gold_data, global_coref_clusters, counts_threshold=2)
        coref_string = "global_coref"

    saliency_rates = list(saliency_dict.values())
    n, bins, patches = plt.hist(saliency_rates, 10, facecolor='green', alpha=0.75)
    plt.savefig("/tmp/saliency_rates_histogram.png")

    predicted_salient_mentions_a_list = [convert_to_dict(load_jsonl(salient_mentions_file)) for salient_mentions_file in args.salient_mentions_files_a]
    predicted_salient_mentions_b_list = [convert_to_dict(load_jsonl(salient_mentions_file)) for salient_mentions_file in args.salient_mentions_files_b]

    bucket_width = 1.0 / args.num_buckets
    prev_bucket_end = 0.0
    saliency_rate_buckets = []
    for i in range(args.num_buckets):
        saliency_rate_buckets.append((prev_bucket_end, prev_bucket_end + bucket_width))
        prev_bucket_end = prev_bucket_end + bucket_width

    for eval_type, eval_type_string in [("f1", "F1"), ("macro_f1", "Macro-F1"), ("fpr", "False Positive Rate"), ("fnr", "False Negative Rate")]:
        print(f"eval_type: {eval_type}")
        bucketed_eval_comparison = salent_mentions_metrics(test_gold_data, predicted_salient_mentions_a_list, predicted_salient_mentions_b_list, saliency_dict, saliency_rate_buckets, eval_type=eval_type)
        draw_box_plot_with_error_bars(bucketed_eval_comparison,
                                    None,
                                    eval_type_string,
                                    fname=f"/tmp/bucketed_salient_mention_eval_graph_{eval_type}_bucketed_on_train_saliency_rate_{args.num_buckets}_{coref_string}.pdf"
                                    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
