import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

from scirex.metrics.paired_bootstrap import eval_with_hierarchical_paired_bootstrap
from scirex.evaluation_scripts.relations_only_evaluate import prepare_data, compute_relations_metrics
from scirex.commands.join_scirex_and_s2orc import S2OrcEntry, S2Metadata, bucket_documents_by_graph_degree
from scirex.evaluation_scripts.relation_bootstrap_comparison_bucketing import draw_box_plot_with_error_bars

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file")
    parser.add_argument("--ner-file")
    parser.add_argument("--clusters-file")

    parser.add_argument("--relations-files-a", help="Relation prediction files from system A", nargs='+', type=str)
    parser.add_argument("--threshes-a", help="List of thresholds to use", nargs='*', type=float)
    parser.add_argument("--relations-files-b", help="Relation predictions files from system B", nargs='+', type=str)
    parser.add_argument("--threshes-b", help="List of thresholds to use", nargs='*', type=float)

    parser.add_argument("--edge-degree-direction", default="both", choices=["both", "out", "in"], type=str)
    parser.add_argument("--metric", default="f1", type=str)
    parser.add_argument("--metric-type", default="retrieval", choices=["retrieval", "classification"], type=str)

    args = parser.parse_args()


    thresholds_a = args.threshes_a
    if thresholds_a is None or len(thresholds_a) == 0:
        thresholds_a = [None] * len(args.relations_files_a)
    else:
        assert len(thresholds_a) == len(args.relations_files_a)

    thresholds_b = args.threshes_b
    if thresholds_b is None or len(thresholds_b) == 0:
        thresholds_b = [None] * len(args.relations_files_b)
    else:
        assert len(thresholds_b) == len(args.relations_files_b)

    processed_datas_a = []
    for rel_file_a in args.relations_files_a:
        processed_data_a = prepare_data(args.gold_file, args.ner_file, args.clusters_file, rel_file_a)
        processed_datas_a.append(processed_data_a)

    processed_datas_b = []
    for rel_file_b in args.relations_files_b:
        processed_data_b = prepare_data(args.gold_file, args.ner_file, args.clusters_file, rel_file_b)
        processed_datas_b.append(processed_data_b)

    gold_data, predicted_ner, predicted_salient_clusters, _, _ = processed_data_b
    cluster_width_buckets = [(0.0, 0.29), (0.29, 0.37), (0.37, 0.4), (0.4, 0.54)]

    for n in [4]:
        bucketed_eval_comparison = {}

        print("\n")
        print(f"n: {n}")
        for cluster_width_bucket in cluster_width_buckets:
            bucket_name_formatted = f"({round(cluster_width_bucket[0], 3)}, {round(cluster_width_bucket[1], 3)})"
            print(f"Bucket Name: {bucket_name_formatted}")
            y_preds_a_list = []
            y_preds_b_list = []
            retrieval_a_list = []
            retrieval_b_list = []
            preds_length = None
            y_labels = None
            retrieval_length = None
            for thresh_a, processed_data_a in zip(thresholds_a, processed_datas_a):
                gold_data_a, predicted_ner_a, predicted_salient_clusters_a, predicted_relations_a, predicted_cluster_to_gold_cluster_map_a = processed_data_a
                assert gold_data_a == gold_data
                assert predicted_ner_a == predicted_ner
                assert predicted_salient_clusters_a == predicted_salient_clusters


                retrieval_metrics_df_a, _, y_labels_a, y_preds_a, _, _, _ = compute_relations_metrics(
                                                        gold_data,
                                                        predicted_ner,
                                                        predicted_salient_clusters,
                                                        predicted_relations_a,
                                                        predicted_cluster_to_gold_cluster_map_a,
                                                        n=n,
                                                        thresh=thresh_a,
                                                        cluster_width_bucket=cluster_width_bucket)
                if y_labels is None:
                    y_labels = y_labels_a
                else:
                    assert y_labels == y_labels_a, breakpoint()

                y_preds_a_list.append(y_preds_a)
                if preds_length is None:
                    preds_length = len(y_preds_a)
                else:
                    assert preds_length == len(y_preds_a)

                if retrieval_length is None:
                    retrieval_length = len(retrieval_metrics_df_a[args.metric])
                else:
                    assert retrieval_length == len(retrieval_metrics_df_a[args.metric])
                retrieval_a_list.append(retrieval_metrics_df_a)

            for thresh_b, processed_data_b in zip(thresholds_b, processed_datas_b):
                gold_data_b, predicted_ner_b, predicted_salient_clusters_b, predicted_relations_b, predicted_cluster_to_gold_cluster_map_b = processed_data_b
                assert gold_data_b == gold_data
                assert predicted_ner_b == predicted_ner
                assert predicted_salient_clusters_b == predicted_salient_clusters

                retrieval_metrics_df_b, _, y_labels_b, y_preds_b, _, _, _ = compute_relations_metrics(
                                                        gold_data,
                                                        predicted_ner,
                                                        predicted_salient_clusters,
                                                        predicted_relations_b,
                                                        predicted_cluster_to_gold_cluster_map_b,
                                                        n=n,
                                                        thresh=thresh_b,
                                                        cluster_width_bucket=cluster_width_bucket)
                assert y_labels == y_labels_b, breakpoint()
                assert preds_length == len(y_preds_b)
                y_preds_b_list.append(y_preds_b)
                assert retrieval_length == len(retrieval_metrics_df_b[args.metric])
                retrieval_b_list.append(retrieval_metrics_df_b)

            print(f"Paired Bootstrap Comparison of System A and System B on relation classification metric:")
            if args.metric  == "f1":
                relation_metric = 'macro-f1'
            elif args.metric == "p":
                relation_metric = 'precision'
            elif args.metric == "r":
                relation_metric = 'recall'
            else:
                raise ValueError(f"Unexpected metric {args.metric} received!")

            sys1_summary_class, sys2_summary_class, p_value_lose_class, p_value_win_class = eval_with_hierarchical_paired_bootstrap(y_labels, y_preds_a_list, y_preds_b_list,
                                    num_samples=5000, sample_ratio=0.50,
                                    eval_type=relation_metric, return_results=True)

            print("\n")
            print(f"Paired Bootstrap Comparison of System A and System B on relation retrieval metric:")
            # The bootstrap script expects a list of gold values, but here the "system" values are already
            # comparisons with gold, so just pass in a list of Nones to satisfy the input.
            metric_type = args.metric
            sys1_retrieval_list = [metric[metric_type] for metric in retrieval_a_list]
            sys2_retrieval_list = [metric[metric_type] for metric in retrieval_b_list]
            assert len(sys1_retrieval_list[0]) == len(sys2_retrieval_list[0])

            gold = [None for _ in sys1_retrieval_list[0]]
            # Each bootstrap sample draws 50 items.
            sys1_summary_ret, sys2_summary_ret, p_value_lose_ret, p_value_win_ret = eval_with_hierarchical_paired_bootstrap(gold, sys1_retrieval_list, sys2_retrieval_list,
                                    num_samples=5000, sample_ratio=0.76,
                                    eval_type='avg', return_results=True)
            #bucketed_eval_comparison[str(bucket_name)] = {"base": [list(sys1_summary_ret), p_value_lose_ret], "diff": [list(sys2_summary_ret), p_value_win_ret]}
            if args.metric_type == "retrieval":
                bucketed_eval_comparison[bucket_name_formatted] = {"base": [list(sys1_summary_ret), p_value_lose_ret], "diff": [list(sys2_summary_ret), p_value_win_ret]}
                ylabel = 'Relation Retrieval F1 score'
            else:
                bucketed_eval_comparison[bucket_name_formatted] = {"base": [list(sys1_summary_class), p_value_lose_class], "diff": [list(sys2_summary_class), p_value_win_class]}
                ylabel = 'Corpus-Level F1'
        print(f"Bucket evaluations (base):\n{json.dumps(bucketed_eval_comparison, indent=2)}")

        xlabel = 'Minimum cluster distance of relation (as a fraction of document length)'
        draw_box_plot_with_error_bars(bucketed_eval_comparison, xlabel, ylabel, fname=f"/tmp/bucketed_eval_graph_and_citance_comparison_bucketed_by_avg_cluster_distance_{args.metric_type}_{len(cluster_width_buckets)}_n_{n}.pdf")


if __name__ == "__main__":
    main()