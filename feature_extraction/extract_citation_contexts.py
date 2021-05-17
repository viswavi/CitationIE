import argparse
from collections import defaultdict
import shutil
import spacy
import gzip
import json
import jsonlines
import os
import numpy as np
import pickle
import random
import time
from tqdm import tqdm
import wget

from feature_extraction.citation_graph_tf_idf import load_full_text_documents
from feature_extraction.save_graph_embedding_matrix import load_embedding_matrix

from join_scirex_and_s2orc import (
    caches_directory,
    get_citation_graph,
    get_scirex_docids,
    get_scirex_neighbor_texts,
    get_scirex_to_s2orc_mappings,
    get_shard_id_from_path  ,
    metadata_download_script,
    S2OrcEntry,
    S2Metadata
)

random_generator = random.Random(0)

'''
This function is adapted from https://github.com/allenai/SciREX/blob/master/dygiepp/scripts/data/ace-event/parse_ace_event.py#L294-L375

Creates a spaCy object which provides rule-based tokenization and sentence segmentation.
The tokenizer treats punctuation (e.g. '[') as unique tokens. This segmenter is slow but seems to match
the tokenization used in the original SciREX dataset.
'''
def make_tok_seg():
    '''
    Add a few special cases to spacy tokenizer so it works with ACe mistakes.
    '''
    # Prevent edge case where there are sentence breaks in bad places
    def custom_seg(doc):
        for index, token in enumerate(doc):
            if '--' in token.text:
                doc[index].sent_start = False
                if index < len(doc) - 1:
                    doc[index + 1].sent_start = False
            # Comma followed by whitespace doesn't end a sentence.
            if token.text == "," and  index < len(doc) - 2 and doc[index + 1].is_space:
                doc[index + 2].sent_start = False
            # "And" only starts a sentence if preceded by period or question mark.
            if token.text in ["and", "but"] and index >= 1 and doc[index - 1].text not in [".", "?", "!"]:
                doc[index].sent_start = False
            if (not ((token.is_punct and token.text not in [",", "_", ";", "...", ":", "(", ")", '"']) or token.is_space)
                and index < len(doc) - 1):
                doc[index + 1].sent_start = False
            if "\n" in token.text:
                if index + 1 < len(doc):
                    next_token = doc[index + 1]
                    if len(token) > 1:
                        next_token.sent_start = True
                    else:
                        next_token.sent_start = False
            if token.text == "-":
                if index > 0 and index < len(doc) - 1:
                    before = doc[index - 1]
                    after = doc[index + 1]
                    if not (before.is_space or before.is_punct or after.is_space or after.is_punct):
                        after.sent_start = False
        return doc

    nlp = spacy.load('en')
    nlp.add_pipe(custom_seg, before='parser')

    single_tokens = ['sgt.',
                        'sen.',
                        'col.',
                        'brig.',
                        'gen.',
                        'maj.',
                        'sr.',
                        'lt.',
                        'cmdr.',
                        'u.s.',
                        'mr.',
                        'p.o.w.',
                        'u.k.',
                        'u.n.',
                        'ft.',
                        'dr.',
                        'd.c.',
                        'mt.',
                        'st.',
                        'snr.',
                        'rep.',
                        'ms.',
                        'capt.',
                        'sq.',
                        'jr.',
                        'ave.']
    for special_case in single_tokens:
        nlp.tokenizer.add_special_case(special_case, [dict(ORTH=special_case)])
        upped = special_case.upper()
        nlp.tokenizer.add_special_case(upped, [dict(ORTH=upped)])
        capped = special_case.capitalize()
        nlp.tokenizer.add_special_case(capped, [dict(ORTH=capped)])

    return nlp

# Construct spaCy sentence segmenter.
nlp = make_tok_seg()

def get_citation_contexts(s2orc_document, scirex_document_s2orc_ids, context_window_size=0):
    # Mapping from scirex ID to sentence(s?) of citance of that document in citing document.
    citation_contexts = {}
    bibliography = s2orc_document['bib_entries']

    for section in s2orc_document['body_text']:
        for citation in section['cite_spans']:
            if citation['ref_id'] not in bibliography:
                continue
            # Check that the citation points to a document in our SciREX set:
            s2orc_paper_id = bibliography[citation['ref_id']]['link']
            if s2orc_paper_id in scirex_document_s2orc_ids:
                if s2orc_paper_id in citation_contexts:
                    # We only want the first context for each time our paper is mentioned in the document.
                    continue
                text_before_citation = section['text'][:citation['start']]
                text_after_citation = section['text'][citation['end']:]
                # Replace the citation number for the SciREX document with a  "[CITE]" token
                reformatted_text = text_before_citation + "[CITE]" + text_after_citation

                sentences_tokenized = [sent for sent in nlp(reformatted_text).sents]
                if len([i for i, sent in enumerate(sentences_tokenized) if "[CITE]" in sent.orth_]) == 0:
                    start_split = [i for i, sent in enumerate(sentences_tokenized) if "CITE]" in sent.orth_]
                    end_split = [i for i, sent in enumerate(sentences_tokenized) if "[CITE" in sent.orth_]
                    if len(start_split) == 1:
                        [ind] = start_split
                        sentences_tokenized[ind-1] = sentences_tokenized[ind-1] + sentences_tokenized[ind]
                        del sentences_tokenized[ind]
                    elif len(end_split) == 1:
                        [ind] = end_split
                        sentences_tokenized[ind+1] = sentences_tokenized[ind] + sentences_tokenized[ind+1]
                        del sentences_tokenized[ind]
                    else:
                        raise ValueError("We're assuming the sentence tokenizer split the cite token before one of the brackets")

                [citance_id] = [i for i, sent in enumerate(sentences_tokenized) if "[CITE]" in sent.orth_]
                start_citance_idx = max(citance_id - context_window_size, 0)
                end_citance_idx = min(citance_id + 1 + context_window_size, len(sentences_tokenized))
                
                context_sentences = sentences_tokenized[start_citance_idx:end_citance_idx]
                context_sentences = [sentence.as_doc() for sentence in context_sentences]
                citation_contexts[s2orc_paper_id] = context_sentences
    return citation_contexts

def check_citation_context_quality(sentence, min_num_tokens=10):
    # TODO: add more checks here to ensure we only keep high-quality citation contexts
    context_string = " ".join([s.text for s in sentence])
    return len(context_string.split()) >= min_num_tokens

def construct_citation_contexts(document_reader, citing_paper_ids, scirex_document_s2orc_ids, context_window_size=0):
    accumulated_citation_contexts = defaultdict(dict)
    for doc in tqdm(document_reader):
        if doc['paper_id'] not in citing_paper_ids:
            # Can skip papers that we know do not cite any SciREX documents
            continue
        doc_citation_contexts = get_citation_contexts(doc,
                                                      scirex_document_s2orc_ids=scirex_document_s2orc_ids,
                                                      context_window_size=context_window_size)
        for scirex_docid, citation_context in doc_citation_contexts.items():
            accumulated_citation_contexts[scirex_docid][doc['paper_id']] = citation_context
    return accumulated_citation_contexts

def construct_graph_embeddings_matrix(citing_paper_ids, document_id_type='scirex'):
    graph_pickle_file_path = "SciREX/graph_embeddings/graph.pkl"
    node_id_mapping_path = "node_id_mapping.json"
    matrix, doc_id_to_matrix_idx_mapping, average_embedding = load_embedding_matrix(graph_pickle_file_path,
                                   node_id_mapping_path,
                                   citing_paper_ids,
                                   document_id_type=document_id_type)
    # Initialize an embedding dictionary with a given default value (we use the average embedding
    # among the provided document ids).
    embedding_dict = defaultdict(lambda: average_embedding)
    for doc_id, matrix_idx in doc_id_to_matrix_idx_mapping.items():
        embedding_dict[doc_id] = matrix[matrix_idx]
    return embedding_dict

def filter_citation_contexts(all_citation_contexts,
                             max_num_contexts_to_keep=20,
                             citance_selection_method='random',
                             scirex_document_embeddings = None,
                             citing_paper_embeddings = None):
    # COPIED
    if citance_selection_method not in ['random', 'graph_embedding_distance']:
        raise ValueError(f"Unsupported citance selection method {citance_selection_method} received")

    citation_context_sentences = {}
    citation_context_sentences_with_docids = {}
    for scirex_docid in all_citation_contexts:
        contexts = all_citation_contexts[scirex_docid]
        if citance_selection_method == "graph_embedding_distance":
            if scirex_docid not in scirex_document_embeddings:
                print(f"Document {scirex_docid} not in document embeddings")
            scirex_document_embedding = scirex_document_embeddings[scirex_docid]
        if len(contexts) > max_num_contexts_to_keep:
            if citance_selection_method == 'graph_embedding_distance':
                citing_document_ids = list(contexts.keys())
                dot_scores = np.array([np.linalg.norm(scirex_document_embedding - citing_paper_embeddings[s2orc_id]) for s2orc_id in citing_document_ids])
                most_similar_citing_document_idxs = np.argsort(-dot_scores)[:max_num_contexts_to_keep]
                citation_context_sentences[scirex_docid] = [contexts[citing_document_ids[idx]] for idx in most_similar_citing_document_idxs]
                # For debugging only
                citation_context_sentences_with_docids[scirex_docid] = {}
                for idx in most_similar_citing_document_idxs:
                    citing_doc_id = citing_document_ids[idx]
                    citation_context_sentences_with_docids[scirex_docid][citing_document_ids[idx]] = [c.text for c in contexts[citing_doc_id]]
            else:
                context_sentences = [sentence for sentence in contexts.values() if check_citation_context_quality(sentence)]
                citation_context_sentences[scirex_docid] = random_generator.sample(context_sentences, max_num_contexts_to_keep)
                citation_context_sentences_with_docids[scirex_docid] = []
                for context in citation_context_sentences[scirex_docid]:
                    citation_context_sentences_with_docids[scirex_docid].append([c.text for c in context])
    json.dump(citation_context_sentences_with_docids, open(f"/tmp/selection_context_sentences_{citance_selection_method}.json", 'w'), indent=2)
    return citation_context_sentences


def augment_rows_with_citation_context(original_file_path, new_file_path, citation_contexts, scirex_to_s2orc_id_mapping):
    reader = jsonlines.open(original_file_path)
    writer = jsonlines.open(new_file_path, 'w')
    for doc in tqdm(reader):
        doc_id = doc['doc_id']
        if doc_id not in scirex_to_s2orc_id_mapping:
            skip_document = True
        else:
            s2orc_paper_id = scirex_to_s2orc_id_mapping[doc_id]
            if s2orc_paper_id not in citation_contexts or len(citation_contexts[s2orc_paper_id]) == 0:
                skip_document = True
            else:
                skip_document = False
                doc_citation_contexts = citation_contexts[s2orc_paper_id]

        index_counter = len(doc['words'])
        if skip_document:
            doc['citances'] = [index_counter, index_counter]
        else:
            # augment `doc` with citation contexts
            # Specifically, add these to:
            #   ✓ words 
            #   ✓ sections
            #   ✓ sentences
            #   - ner/coref? (not for now)

            words = []
            section_boundaries = []
            sentence_boundaries = []

            # Start the index counter at the last word in the un-augmented text.
            citances_start = index_counter
            for section in doc_citation_contexts:
                section_start = index_counter
                for sentence in section:
                    sentence_start = index_counter
                    for word in sentence:
                        words.append(word.orth_)
                        index_counter += 1
                    sentence_end = index_counter
                    sentence_boundaries.append([sentence_start, sentence_end])
                section_end = index_counter
                section_boundaries.append([section_start, section_end])
            citances_end = index_counter

            #section_boundaries = []
            #section_boundaries.append([citances_start, citances_end])

            doc['words'].extend(words)
            doc['sentences'].extend(sentence_boundaries)
            doc['sections'].extend(section_boundaries)
            doc['citances'] = [citances_start, citances_end]

        writer.write(doc)

def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_documents_to_load',
                            type=int,
                            default=None,
                            help="If set, we will only extract citation contexts from a limited number of documents.")
    parser.add_argument('--input_scirex_data_directory',
                            type=str,
                            default="SciREX/scirex_dataset/release_data/",
                            help="If set, we will only extract citation contexts from a limited number of documents.")
    parser.add_argument('--new_scirex_data_directory',
                            type=str,
                            default="SciREX/scirex_dataset/data_with_citances/", 
                            help="If set, we will only extract citation contexts from a limited number of documents.")
    parser.add_argument('--max_number_of_document_citances',
                            type=int,
                            default=20,
                            help="Limit on the number of citances to write for each document")
    parser.add_argument('--citance_sorting_style',
                            type=str,
                            choices=["random", "graph_embedding_distance"],
                            default="random",
                            help="When a paper has more than max_number_of_document_citances citances, the " + \
                                "requested method is used to select which document citations to include")
    args = parser.parse_args()

    # Load citation graph
    scirex_paper_ids = list(set(get_scirex_docids()))
    scirex_to_s2orc_id_mapping = get_scirex_to_s2orc_mappings()
    scirex_s2orc_ids = set([scirex_to_s2orc_id_mapping[p] for p in scirex_paper_ids if p in scirex_to_s2orc_id_mapping])

    full_citation_contexts_cache_path = os.path.join(caches_directory, "citation_contexts.pkl")
    _, in_citations = get_citation_graph(1)
    citing_paper_ids = list(set([node for neighbor_set in in_citations.values() for node in neighbor_set]))

    if args.citance_sorting_style == "graph_embedding_distance":
        print("Constructing graph embeddings matrices.")
        scirex_document_embeddings = construct_graph_embeddings_matrix(list(scirex_s2orc_ids), document_id_type='s2orc')
        citing_paper_embeddings = construct_graph_embeddings_matrix(citing_paper_ids, document_id_type='s2orc')
    else:
        scirex_document_embeddings = None
        citing_paper_embeddings = None


    if os.path.exists(full_citation_contexts_cache_path):
        print("Unpickling citation contexts cache.")
        unpickling_start = time.perf_counter()
        citation_contexts = pickle.load(open(full_citation_contexts_cache_path, 'rb'))
        unpickling_end = time.perf_counter()
        print(f"Unpickling citation contexts cache took {unpickling_end - unpickling_start} seconds")
    else:
        print("Manually constructing citation contexts (will take ~2 hours).")
        _, scigraph_documents_path = get_scirex_neighbor_texts()
        document_reader = load_full_text_documents(scigraph_documents_path, num_documents_to_load=args.num_documents_to_load)

        print("Constructing citation contexts")
        citation_contexts = construct_citation_contexts(document_reader, citing_paper_ids, scirex_s2orc_ids, context_window_size=1)
        pickle.dump(citation_contexts, open(full_citation_contexts_cache_path, 'wb'))


    citation_contexts_truncated = filter_citation_contexts(citation_contexts,
                                                           max_num_contexts_to_keep = args.max_number_of_document_citances,
                                                           citance_selection_method = args.citance_sorting_style,
                                                           scirex_document_embeddings = scirex_document_embeddings,
                                                           citing_paper_embeddings = citing_paper_embeddings)
    #citation_contexts_truncated_file = os.path.join(caches_directory, "citation_contexts_truncated.pkl")
    #pickle.dump(citation_contexts_truncated, open(citation_contexts_truncated_file, 'wb'))

    #citation_contexts_truncated_file = os.path.join(caches_directory, "citation_contexts_truncated.pkl")
    #citation_contexts_truncated = pickle.load(open(citation_contexts_truncated_file, 'rb'))

    os.makedirs(args.new_scirex_data_directory, exist_ok=True)
    shutil.copyfile(os.path.join(args.input_scirex_data_directory, "dev.jsonl"),
                    os.path.join(args.new_scirex_data_directory, "dev.jsonl"))

    print(f"Writing dataset splits with citation contexts")
    print("Train")
    augment_rows_with_citation_context(
                    os.path.join(args.input_scirex_data_directory, "train.jsonl"),
                    os.path.join(args.new_scirex_data_directory, "train.jsonl"),
                    citation_contexts_truncated,
                    scirex_to_s2orc_id_mapping)
    print("Test")
    #shutil.copyfile(os.path.join(args.input_scirex_data_directory, "test.jsonl"),
    #                os.path.join(args.new_scirex_data_directory, "test.jsonl"))
    augment_rows_with_citation_context(
                    os.path.join(args.input_scirex_data_directory, "test.jsonl"),
                    os.path.join(args.new_scirex_data_directory, "test.jsonl"),
                    citation_contexts_truncated,
                    scirex_to_s2orc_id_mapping)

    end = time.perf_counter()
    print(f"Script took {end - start} seconds to run.")


    

if __name__ == "__main__":
    main()