import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

from scirex.metrics.paired_bootstrap import eval_with_paired_bootstrap
from scirex.evaluation_scripts.relations_only_evaluate import prepare_data, compute_relations_metrics
from scirex.commands.join_scirex_and_s2orc import S2OrcEntry, S2Metadata, bucket_documents_by_graph_degree

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
    ax.bar(r2, diff_means, color='lightblue', width=barWidth, edgecolor='white', label='w/ Graph + Citance', yerr=diff_error_bars, error_kw=error_kw)

    # Add xticks on the middle of the group bars
    # ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(bucket_names, rotation=15)
    ax.set_xticks([r + barWidth for r in range(len(r1))])

    # Create legend & Show graphic
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(fname, dpi=400, transparent=True)
    print(f"Wrote plot to {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file")
    parser.add_argument("--ner-file")
    parser.add_argument("--clusters-file")
    parser.add_argument("--relations-file-a", help="Relation predictions from system A")
    parser.add_argument("--thresh-a", default=None, type=float)
    parser.add_argument("--relations-file-b", help="Relation predictions from system A")
    parser.add_argument("--thresh-b", default=None, type=float)
    parser.add_argument("--edge-degree-direction", default="both", choices=["both", "out", "in"], type=str)
    parser.add_argument("--num-buckets", default=6, type=int)
    parser.add_argument("--metric-type", default="retrieval", choices=["retrieval", "classification"], type=str)

    args = parser.parse_args()

    processed_data_a = prepare_data(args.gold_file, args.ner_file, args.clusters_file, args.relations_file_a)
    gold_data_a, predicted_ner_a, predicted_salient_clusters_a, predicted_relations_a, predicted_cluster_to_gold_cluster_map_a = processed_data_a

    processed_data_b = prepare_data(args.gold_file, args.ner_file, args.clusters_file, args.relations_file_b)
    gold_data_b, predicted_ner_b, predicted_salient_clusters_b, predicted_relations_b, predicted_cluster_to_gold_cluster_map_b = processed_data_b

    assert gold_data_a == gold_data_b
    gold_data = gold_data_a
    assert predicted_ner_a == predicted_ner_b
    predicted_ner = predicted_ner_a
    assert predicted_salient_clusters_a == predicted_salient_clusters_b
    predicted_salient_clusters = predicted_salient_clusters_a

    doc_ids = [doc["doc_id"] for doc in gold_data]
    doc_buckets = bucket_documents_by_graph_degree(doc_ids, num_buckets=args.num_buckets, degree_direction=args.edge_degree_direction)

    for n in [4]:
        bucketed_eval_comparison = {}
        bucket_evaluations_diff = {}

        print("\n")
        print(f"n: {n}")
        for bucket_name, bucket_docs in doc_buckets:
            gold_data_in_bucket = [doc for doc in gold_data if doc["doc_id"] in bucket_docs]
            print("\n")
            print(f"bucket: {bucket_name}, contains {len(gold_data_in_bucket)} documents")
            retrieval_metrics_df_a, _, y_labels_a, y_preds_a, _, _, _ = compute_relations_metrics(
                                                    gold_data_in_bucket,
                                                    predicted_ner,
                                                    predicted_salient_clusters,
                                                    predicted_relations_a,
                                                    predicted_cluster_to_gold_cluster_map_a,
                                                    n=n,
                                                    thresh=args.thresh_a)
            retrieval_metrics_df_b, _, y_labels_b, y_preds_b, _, _, _ = compute_relations_metrics(
                                                    gold_data_in_bucket,
                                                    predicted_ner,
                                                    predicted_salient_clusters,
                                                    predicted_relations_b,
                                                    predicted_cluster_to_gold_cluster_map_b,
                                                    n=n,
                                                    thresh=args.thresh_b)
            assert y_labels_a == y_labels_b, breakpoint()
            y_labels = y_labels_a
            assert len(y_preds_a) == len(y_preds_b)

            print(f"Paired Bootstrap Comparison of System A and System B on relation classification metric:")
            sys1_summary_class, sys2_summary_class, p_value_lose_class, p_value_win_class = eval_with_paired_bootstrap(y_labels, y_preds_a, y_preds_b,
                                    num_samples=5000, sample_ratio=0.50,
                                    eval_type='macro-f1', return_results=True)

            print("\n")
            print(f"Paired Bootstrap Comparison of System A and System B on relation retrieval metric:")
            # The bootstrap script expects a list of gold values, but here the "system" values are already 
            # comparisons with gold, so just pass in a list of Nones to satisfy the input.
            sys1_retrieval = list(retrieval_metrics_df_a["f1"])
            sys2_retrieval = list(retrieval_metrics_df_b["f1"])
            assert len(sys1_retrieval) == len(sys2_retrieval)

            gold = [None for _ in sys1_retrieval]
            # Each bootstrap sample draws 50 items.
            sys1_summary_ret, sys2_summary_ret, p_value_lose_ret, p_value_win_ret = eval_with_paired_bootstrap(gold, sys1_retrieval, sys2_retrieval,
                                    num_samples=1000, sample_ratio=0.75,
                                    eval_type='avg', return_results=True)
            if args.metric_type == "retrieval":
                bucketed_eval_comparison[str(bucket_name)] = {"base": [list(sys1_summary_ret), p_value_lose_ret], "diff": [list(sys2_summary_ret), p_value_win_ret]}
            else:
                bucketed_eval_comparison[str(bucket_name)] = {"base": [list(sys1_summary_class), p_value_lose_class], "diff": [list(sys2_summary_class), p_value_win_class]}
        print(f"Bucket evaluations (base):\n{json.dumps(bucketed_eval_comparison, indent=2)}")

        draw_box_plot_with_error_bars(bucketed_eval_comparison, 'Degree of documents in citation graph)', 'Mean Retrieval F1 score', fname=f"/tmp/bucketed_eval_comparison_bucket_{args.metric_type}_{args.num_buckets}_n_{n}.png")


if __name__ == "__main__":
    main()