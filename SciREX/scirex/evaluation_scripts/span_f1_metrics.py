from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
        bio_tags_to_spans,
        bioul_tags_to_spans,
        iob1_tags_to_spans,
        bmes_tags_to_spans,
        TypedStringSpan
)


TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]


class SpanBasedF1Measure(Metric):
    """
    Copied (Span Based F1 Measure from allennlp)

    """
    def __init__(self, entity_labels) -> None:
        # These will hold per label span counts.
        self._entity_labels = entity_labels
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    @staticmethod
    def convert_list_to_tuples(entity_list):
        type_dict = defaultdict(list)
        for [span_start, span_end, entity] in entity_list:
            type_dict[entity].append((span_start, span_end))
        return type_dict

    def __call__(self,
                 predictions: list,
                 gold_labels: list):
        typed_predicted_spans = self.convert_list_to_tuples(predictions)
        typed_gold_spans = self.convert_list_to_tuples(gold_labels)

        for label in typed_predicted_spans :
            for p_span in typed_predicted_spans[label] :
                matched = False
                for g_span in typed_gold_spans[label] :
                    if exact_match(p_span, g_span) == 1.0 :
                        self._true_positives[label] += 1
                        matched = True
                        typed_gold_spans[label].remove(g_span)
                        break
                if not matched :
                    self._false_positives[label] += 1

        for label in typed_gold_spans :
            for span in typed_gold_spans[label] :
                self._false_negatives[label] += 1

        # for span in predicted_spans:
        #     if span in gold_spans:
        #         self._true_positives[span[0]] += 1
        #         gold_spans.remove(span)
        #     else:
        #         self._false_positives[span[0]] += 1
        # # These spans weren't predicted.
        # for span in gold_spans:
        #     self._false_negatives[span[0]] += 1

    def get_metric(self, reset: bool = False):
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

def span_match(span_1, span_2) :
    sa, ea = span_1
    sb, eb = span_2
    ea, eb = ea + 1, eb + 1
    iou = (min(ea, eb) - max(sa, sb)) / (max(eb, ea) - min(sa, sb))
    return iou

def exact_match(span_1, span_2) :
    return float(span_1 == span_2)