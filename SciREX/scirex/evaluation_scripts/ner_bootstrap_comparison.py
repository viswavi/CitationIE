import argparse
from itertools import combinations
import json
from typing import Dict

import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score
from span_f1_metrics import SpanBasedF1Measure
from scirex.metrics.paired_bootstrap import eval_with_paired_bootstrap
from scirex.metrics.clustering_metrics import match_predicted_clusters_to_gold
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex_utilities.json_utilities import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--ner-file-a")
parser.add_argument("--ner-file-b")


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


def main(args):
    gold_data = load_jsonl(args.gold_file)
    all_label_ner_values = [x['ner'] for x in gold_data]
    entity_labels = set([entity[2] for document in all_label_ner_values for entity in document])

    f1metric_a = SpanBasedF1Measure(entity_labels=entity_labels)
    f1metric_b = SpanBasedF1Measure(entity_labels=entity_labels)
    predicted_ner_a = convert_to_dict(load_jsonl(args.ner_file_a))
    predicted_ner_b = convert_to_dict(load_jsonl(args.ner_file_b))

    assert len(gold_data) == len(predicted_ner_a)
    assert len(gold_data) == len(predicted_ner_b)

    single_y_labels = None
    y_preds_a = []
    y_labels_a = []
    y_preds_b = []
    y_labels_b = []
    for labeled_doc in gold_data:
        labeled_ner_values = labeled_doc['ner']
        doc_id = labeled_doc['doc_id']

        f1metric_a.reset()
        f1metric_a(predicted_ner_a[doc_id]['ner'], labeled_ner_values)
        
        tps = sum(f1metric_a._true_positives.values())
        fps = sum(f1metric_a._false_positives.values())
        fns = sum(f1metric_a._false_negatives.values())

        for _ in range(tps):
            y_preds_a.append(1)
            y_labels_a.append(1)

        for _ in range(fns):
            y_preds_a.append(0)
            y_labels_a.append(1)

        for _ in range(fps):
            y_preds_a.append(1)
            y_labels_a.append(0)
        

        f1metric_b.reset()
        f1metric_b(predicted_ner_b[doc_id]['ner'], labeled_ner_values)
        
        tps = sum(f1metric_b._true_positives.values())
        fps = sum(f1metric_b._false_positives.values())
        fns = sum(f1metric_b._false_negatives.values())

        for _ in range(tps):
            y_preds_b.append(1)
            y_labels_b.append(1)

        for _ in range(fns):
            y_preds_b.append(0)
            y_labels_b.append(1)

        for _ in range(fps):
            y_preds_b.append(1)
            y_labels_b.append(0)

        # Add TNs to round things out
        if len(y_labels_b) > len(y_labels_a):
            diff = len(y_labels_b) - len(y_labels_a)
            for _ in range(diff):
                y_labels_a.append(0)
                y_preds_a.append(0)
        elif len(y_labels_a) > len(y_labels_b):
            diff = len(y_labels_a) - len(y_labels_b)
            for _ in range(diff):
                y_labels_b.append(0)
                y_preds_b.append(0)

    assert y_labels_a == y_labels_b, breakpoint()
    
    print(f"\nBaseline:")
    print(f"Overall F1: {f1_score(y_labels_a, y_preds_a)}")
    print(f"Overall Precision: {precision_score(y_labels_a, y_preds_a)}")
    print(f"Overall Recall: {recall_score(y_labels_a, y_preds_a)}")
    print(f"\nNew System:")
    print(f"Overall F1: {f1_score(y_labels_b, y_preds_b)}")
    print(f"Overall Precision: {precision_score(y_labels_b, y_preds_b)}")
    print(f"Overall Recall: {recall_score(y_labels_b, y_preds_b)}")

    print(f"Bootstrap (F1)")
    eval_with_paired_bootstrap(y_labels_a, y_preds_a, y_preds_b, num_samples=1000, sample_ratio=0.5, eval_type='f1')

    print(f"\nBootstrap (Precision)")
    eval_with_paired_bootstrap(y_labels_a, y_preds_a, y_preds_b, num_samples=1000, sample_ratio=0.5, eval_type='precision')

    print(f"\nBootstrap (Recall)")
    eval_with_paired_bootstrap(y_labels_a, y_preds_a, y_preds_b, num_samples=1000, sample_ratio=0.5, eval_type='recall')
    



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)