import argparse
from itertools import combinations
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

import pandas as pd

# from scirex.metrics.clustering_metrics import match_predicted_clusters_to_gold
from scirex.commands.join_scirex_and_s2orc import S2OrcEntry, S2Metadata, bucket_documents_by_graph_degree
from scirex.metrics.f1 import compute_f1
from scirex.metrics.paired_bootstrap import eval_with_paired_bootstrap
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex.predictors.utils import map_and_intersect_predicted_clusters_to_gold
from scirex_utilities.entity_utils import used_entities
from scirex_utilities.json_utilities import load_jsonl

def draw_box_plot_with_error_bars(bucketed_eval_comparison, xlabel, ylabel, fname = "/tmp/bucketed_eval_comparison.png"):
    # set width of bar
    barWidth = 0.32

    # set height of bar
    base_means = []
    base_CIs = []
    diff_means = []
    diff_CIs = []

    base_yerr_lower = []
    base_yerr_upper = []
    diff_yerr_lower = []
    diff_yerr_upper = []
    bucket_names = list(bucketed_eval_comparison.keys())
    for bucket in bucketed_eval_comparison.values():
        base_mean = bucket["base"][0][0]
        base_means.append(base_mean)
        base_ci = bucket["base"][0][1]
        base_CIs.append(base_ci)
        base_yerr_lower.append(base_mean - base_ci[0])
        base_yerr_upper.append(base_ci[1] - base_mean)

        diff_mean = bucket["diff"][0][0]
        diff_means.append(diff_mean)
        diff_ci = bucket["diff"][0][1]
        diff_CIs.append(diff_ci)
        diff_yerr_lower.append(diff_mean - diff_ci[0])
        diff_yerr_upper.append(diff_ci[1] - diff_mean)
    base_error_bars=[base_yerr_lower, base_yerr_upper]
    diff_error_bars=[diff_yerr_lower, diff_yerr_upper]

    # Set position of bar on X axis
    r1 = np.arange(len(base_means))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    error_kw=dict(lw=1, capsize=2, capthick=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(r1, base_means, color='peachpuff', width=barWidth, edgecolor='white', label='Baseline', yerr=base_error_bars, error_kw=error_kw)
    ax.bar(r2, diff_means, color='lightblue', width=barWidth, edgecolor='white', label='w/ Graph', yerr=diff_error_bars, error_kw=error_kw)

    # Add xticks on the middle of the group bars
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(bucket_names, rotation=15)
    ax.set_xticks([r + barWidth for r in range(len(r1))])

    # Create legend & Show graphic
    ax.legend(loc='best')
    plt.tight_layout()
    fig.savefig(fname, dpi=400, transparent=True)
    print(f"Wrote plot to {fname}")
    import tikzplotlib
    tikzplotlib.save(f"{fname}.tex")
    print(f"Wrote tikz plot to {fname}.tex")

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

def match_predicted_clusters_to_gold(
    predicted_clusters: Dict[str, List[Tuple[int, int]]],
    gold_clusters: Dict[str, List[Tuple[int, int]]],
    span_map,
    words
):
    intersection_scores = map_and_intersect_predicted_clusters_to_gold(predicted_clusters, gold_clusters, span_map)
    matched_clusters = {}
    
    for p in intersection_scores :
        if len(intersection_scores[p]) > 0:
            g, v = max(list(intersection_scores[p].items()), key=lambda x : x[1])
            if v > 0.5 :
                matched_clusters[p] = g

    metrics = {'p' : len(matched_clusters) / (len(predicted_clusters) + 1e-7), 'r' : len(set(matched_clusters.values())) / (len(gold_clusters) + 1e-7)}
    metrics['f1'] = 2 * metrics['p'] * metrics['r'] / (metrics['p'] + metrics['r'] + 1e-7)

    return metrics, matched_clusters


def salent_mentions_metrics(gold_data, predicted_salient_mentions):
    all_metrics = []
    predicted = 0
    gold = 0
    matched = 0

    preds = []
    labels = []
    for doc in gold_data:
        gold_salient_spans = [span for coref_cluster in doc['coref'].values() for span in coref_cluster]

        predicted_doc = predicted_salient_mentions[doc["doc_id"]]
        saliency_spans = []
        for [start_span, end_span, saliency, _] in predicted_doc["saliency"]:
            preds.append(saliency)
            labels.append((start_span, end_span) in gold_salient_spans)
            if saliency:
                saliency_spans.append((start_span, end_span))

        matching_spans = set(gold_salient_spans).intersection(saliency_spans)
        matched += len(matching_spans)
        predicted += len(saliency_spans)
        gold += len(gold_salient_spans)

    precision, recall, f1 = compute_f1(predicted, gold, matched, m=1)
    all_metrics = pd.DataFrame({"f1": [f1], "p": [precision], "r": [recall]})
    print("Salient Mention Classification Metrics")
    print(all_metrics.describe().loc['mean'])
    return preds, labels


def clustering_metrics(gold_data, predicted_clusters, span_map):
    all_metrics = []
    mappings = {}
    for doc in gold_data:
        predicted_doc = predicted_clusters[doc["doc_id"]]
        metrics, mapping = match_predicted_clusters_to_gold(
            predicted_doc["clusters"], doc["coref"], span_map[doc["doc_id"]], doc['words']
        )
        mappings[doc["doc_id"]] = mapping
        all_metrics.append(metrics)

    all_metrics = pd.DataFrame(all_metrics)
    print("Salient Clustering Metrics")
    print(all_metrics.describe().loc['mean'])

    return mappings, all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file")
    parser.add_argument("--ner-file")
    parser.add_argument("--clusters-file-a", help="Cluster predictions from system A")
    parser.add_argument("--salient-mentions-file-a", help="Salient mentions from system A")
    parser.add_argument("--clusters-file-b", help="Cluster predictions from system B")
    parser.add_argument("--salient-mentions-file-b", help="Salient mentions from system B")
    parser.add_argument("--edge-degree-direction", default="both", choices=["both", "out", "in"], type=str)
    parser.add_argument("--num-buckets", default=6, type=int)
    args = parser.parse_args()

    bucketed_eval_comparison = {}

    gold_data = load_jsonl(args.gold_file)
    for d in gold_data:
        merge_method_subrelations(d)
        d["clusters"] = d["coref"]
    predicted_ner = convert_to_dict(load_jsonl(args.ner_file))
    predicted_span_to_gold_span_map: Dict[str, Dict[tuple, tuple]] = ner_metrics(gold_data, predicted_ner)

    doc_ids = [doc["doc_id"] for doc in gold_data]
    doc_buckets = bucket_documents_by_graph_degree(doc_ids, num_buckets=args.num_buckets)

    for bucket_name, bucket_docs in doc_buckets:
        gold_data_in_bucket = [doc for doc in gold_data if doc["doc_id"] in bucket_docs]
        print("\n")
        print(f"bucket: {bucket_name}, contains {len(gold_data_in_bucket)} documents")

        predicted_salient_mentions_a = convert_to_dict(load_jsonl(args.salient_mentions_file_a))
        preds_a, labels_a = salent_mentions_metrics(gold_data_in_bucket, predicted_salient_mentions_a)

        predicted_salient_mentions_b = convert_to_dict(load_jsonl(args.salient_mentions_file_b))
        preds_b, labels_b = salent_mentions_metrics(gold_data_in_bucket, predicted_salient_mentions_b)
        assert labels_a == labels_b
        gold_mentions = labels_a

        print(f"Paired Bootstrap Comparison of System A and System B on salient mention metric:")
        # The bootstrap script expects a list of gold values, but here the "system" values are already 
        # comparisons with gold, so just pass in a list of Nones to satisfy the input.
        assert len(preds_a) == len(preds_b)
        assert len(preds_a) == len(gold_mentions)
        sys1_mention = list(preds_a)
        sys2_mention = list(preds_b)
        assert len(sys1_mention) == len(sys2_mention)
        sys1_summary, sys2_summary, p_value_lose, p_value_win =eval_with_paired_bootstrap(gold_mentions, sys1_mention, sys2_mention,
                                num_samples=1000, sample_ratio=0.5,
                                eval_type='f1', return_results=True)
        bucketed_eval_comparison[str(bucket_name)] = {"base": [list(sys1_summary), p_value_lose], "diff": [list(sys2_summary), p_value_win]}

        predicted_salient_clusters_a = convert_to_dict(load_jsonl(args.clusters_file_a))
        predicted_salient_clusters_b = convert_to_dict(load_jsonl(args.clusters_file_b))

        get_types_of_clusters(convert_to_dict(gold_data_in_bucket), convert_to_dict(gold_data_in_bucket))

        i = 0
        filenames = [args.salient_mentions_file_a, args.salient_mentions_file_b]
        for predicted_salient_clusters in [predicted_salient_clusters_a, predicted_salient_clusters_b]:
            print(f"\nMetrics for {filenames[i]}")
            i+=1
            for d, doc in predicted_salient_clusters.items() :
                if 'clusters' not in doc :
                    merge_method_subrelations(doc)
                    doc['clusters'] = {x:v for x, v in doc['coref'].items() if len(v) > 0}
            get_types_of_clusters(predicted_ner, predicted_salient_clusters)

        _, all_metrics_a = clustering_metrics(
            gold_data_in_bucket, predicted_salient_clusters_a, predicted_span_to_gold_span_map
        )
        _, all_metrics_b = clustering_metrics(
            gold_data_in_bucket, predicted_salient_clusters_b, predicted_span_to_gold_span_map
        )

        print(f"Paired Bootstrap Comparison of System A and System B on salient cluster metric:")
        # The bootstrap script expects a list of gold values, but here the "system" values are already 
        # comparisons with gold, so just pass in a list of Nones to satisfy the input.
        sys1_cluster = list(all_metrics_a["f1"])
        sys2_cluster = list(all_metrics_b["f1"])
        assert len(sys1_cluster) == len(sys2_cluster)

        gold = [None for _ in sys1_cluster]
        # Each bootstrap sample draws 50 items.
        eval_with_paired_bootstrap(gold, sys1_cluster, sys2_cluster,
                                num_samples=1000, sample_ratio=0.76,
                                eval_type='avg')
    print(f"Bucket evaluations (diff):\n{json.dumps(bucketed_eval_comparison, indent=2)}")
    draw_box_plot_with_error_bars(bucketed_eval_comparison, fname=f"/tmp/bucketed_salient_mention_eval_comparison_n_{args.num_buckets}.png")

if __name__ == "__main__":
    main()