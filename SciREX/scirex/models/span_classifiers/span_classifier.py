import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from overrides import overrides

import numpy as np

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from scirex.metrics.thresholding_f1_metric import BinaryThresholdF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SpanClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        mention_feedforward: FeedForward,
        label_namespace: str,
        n_features: int = 0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        document_embedding: torch.nn.Embedding = None,
        doc_to_idx_mapping: dict = None,
        graph_embedding_dim: int = None,
    ) -> None:
        super(SpanClassifier, self).__init__(vocab, regularizer)
        self._label_namespace = label_namespace

        self._mention_feedforward = TimeDistributed(mention_feedforward)
        self._use_graph_embeddings = graph_embedding_dim is not None

        features_dim = n_features + graph_embedding_dim if self._use_graph_embeddings else n_features
        self._ner_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim() + features_dim, 1))
        self._ner_metrics = BinaryThresholdF1()
        self._document_embedding = document_embedding
        self._doc_to_idx_mapping = doc_to_idx_mapping

        initializer(self)

    '''
    self._saliency_classifier(
        spans=spans,
        span_embeddings=featured_span_embeddings,
        span_features=output_span_embedding["span_features"],
        span_labels=span_saliency_labels,
        metadata=metadata,
    )
    '''
    @overrides
    def forward(
        self,  # type: ignore
        spans: torch.IntTensor,  # (Batch Size, Number of Spans, 2)
        span_embeddings: torch.IntTensor,  # (Batch Size, Number of Spans, Span Embedding SIze)
        span_features: torch.FloatTensor = None,
        span_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Shape: (Batch_size, Number of spans, H)

        span_feedforward = self._mention_feedforward(span_embeddings)

        if span_features is not None :
            span_feedforward = torch.cat([span_feedforward, span_features], dim=-1)
        
        if self._document_embedding is not None:
            document_idxs = torch.tensor([self._doc_to_idx_mapping[meta["doc_id"]] for meta in metadata], device=spans.device)
            graph_features = self._document_embedding(document_idxs)
            (batch_size, num_spans, _) = span_embeddings.shape
            graph_features = graph_features.repeat(1, num_spans).view(batch_size, num_spans, -1)
            span_feedforward = torch.cat([span_feedforward, graph_features], dim=-1)

        ner_scores = self._ner_scorer(span_feedforward).squeeze(-1) #(B, NS)
        ner_probs = torch.sigmoid(ner_scores)

        output_dict = {
            "spans" : spans,
            "ner_probs": ner_probs,
            "loss" : 0.0
        }

        if span_labels is not None:
            assert ner_probs.shape == span_labels.shape, breakpoint()
            assert len(ner_probs.shape) == 2, breakpoint()
            self._ner_metrics(ner_probs, span_labels)
            loss = self._compute_loss_for_scores(ner_probs, span_labels, metadata)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["metadata"] = metadata

        return output_dict

    def _compute_loss_for_scores(self, ner_probs, ner_labels, metadata):
        ner_probs_flat = ner_probs.view(-1)
        ner_labels_flat = ner_labels.view(-1)

        loss = torch.nn.BCELoss(reduction="mean")(ner_probs_flat, ner_labels_flat.float())
        return loss

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        output_dict['decoded_spans'] = []
        if 'spans' in output_dict :
            for spans, spans_prob in zip(output_dict['spans'], output_dict['ner_probs']) :
                decoded = {(span[0].item(), span[1].item() + 1): label.item() for span, label in zip(spans, spans_prob)}
                output_dict['decoded_spans'].append(decoded)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._ner_metrics.get_metric(reset)
        metrics = {"span_" + k: v for k, v in metrics.items()}

        return metrics

