import argparse
from itertools import combinations
import json
from typing import Dict

import pandas as pd

from span_f1_metrics import SpanBasedF1Measure
from scirex.metrics.clustering_metrics import match_predicted_clusters_to_gold
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex_utilities.json_utilities import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--ner-file")


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

    f1metric = SpanBasedF1Measure(entity_labels=entity_labels)
    predicted_ner = convert_to_dict(load_jsonl(args.ner_file))

    assert len(gold_data) == len(predicted_ner)

    for labeled_doc in gold_data:
        labeled_ner_values = labeled_doc['ner']
        doc_id = labeled_doc['doc_id']
        if doc_id not in predicted_ner:
            raise ValueError(f"No predictions found for document {doc_id}")
        predicted_ner_values = predicted_ner[doc_id]['ner']
        f1metric(predicted_ner_values, labeled_ner_values)

    print(json.dumps(f1metric.get_metric(), indent=2))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)