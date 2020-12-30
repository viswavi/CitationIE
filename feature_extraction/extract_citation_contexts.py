import argparse
from collections import defaultdict
import shutil
import spacy
import gzip
import jsonlines
import os
import pickle
import random
import time
import wget

from citation_graph_tf_idf import load_full_text_documents
from join_scirex_and_s2orc import (
    caches_directory,
    get_citation_graph,
    get_scirex_docids,
    get_scirex_neighbor_texts,
    get_scirex_to_s2orc_mappings,
    get_shard_id_from_path,
    metadata_download_script,
    S2OrcEntry,
    S2Metadata
)

from spacy.lang.en import English

random_generator = random.Random(0)

# This is a special character that does not occur in any document
SPECIAL_CHARACTER = chr(10000)

'''
This function is adapted from https://github.com/allenai/SciREX/blob/master/dygiepp/scripts/data/ace-event/parse_ace_event.py#L294-L375

Creates a spaCy object which provides rule-based tokenization and sentence segmentation.
The tokenizer treats punctuation (e.g. '[') as unique tokens.
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
                        from IPython import embed; embed(); raise ValueError("break")
                        raise ValueError("We're assuming the sentence tokenizer split the cite token before one of the brackets")

                try:
                    [citance_id] = [i for i, sent in enumerate(sentences_tokenized) if "[CITE]" in sent.orth_]
                except:
                    from IPython import embed; embed(); raise ValueError("break")
                start_citance_idx = max(citance_id - context_window_size, 0)
                end_citance_idx = min(citance_id + 1 + context_window_size, len(sentences_tokenized))
                
                context_sentences = sentences_tokenized[start_citance_idx:end_citance_idx]
                citation_contexts[s2orc_paper_id] = context_sentences
    return citation_contexts

def check_citation_context_quality(sentence, min_num_tokens=10):
    # TODO: add more checks here to ensure we only keep high-quality citation contexts
    context_string = " ".join([s.orth_ for s in sentence])
    return len(context_string.split()) >= min_num_tokens

def construct_citation_contexts(document_reader, citing_paper_ids, scirex_document_s2orc_ids, context_window_size=0, max_num_contexts_to_keep=20):
    accumulated_citation_contexts = defaultdict(list)
    i = 0
    for doc in document_reader:
        i += 1
        if i > 100:
            break



        if doc['paper_id'] not in citing_paper_ids:
            # Can skip papers that we know do not cite any SciREX documents
            continue
        doc_citation_contexts = get_citation_contexts(doc,
                                                      scirex_document_s2orc_ids=scirex_document_s2orc_ids,
                                                      context_window_size=context_window_size)
        for scirex_docid, citation_context in doc_citation_contexts.items():
            accumulated_citation_contexts[scirex_docid].append(citation_context)

    for scirex_docid in accumulated_citation_contexts:
        contexts = accumulated_citation_contexts[scirex_docid]
        contexts = [sentence for sentence in contexts if check_citation_context_quality(sentence)]
        if len(contexts) > max_num_contexts_to_keep:
            accumulated_citation_contexts[scirex_docid] = random_generator.sample(contexts, max_num_contexts_to_keep)
    return accumulated_citation_contexts


def augment_rows_with_citation_context(original_file_path, new_file_path, citation_contexts, scirex_to_s2orc_id_mapping):
    reader = jsonlines.open(original_file_path)
    writer = jsonlines.open(new_file_path, 'w')
    for doc in reader:
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

        if not skip_document:
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
            index_counter = len(doc['words'])
            for section in doc_citation_contexts:
                section_start = index_counter
                for sentence in section:
                    sentence_start = index_counter
                    for word in sentence:
                        words.append(word)
                        index_counter += 1
                    sentence_end = index_counter
                    sentence_boundaries.append([sentence_start, sentence_end])
                section_end = index_counter
                section_boundaries.append([section_start, section_end])
            doc['words'].extend(words)
            doc['sentences'].extend(sentence_boundaries)
            doc['sections'].extend(section_boundaries)

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
                            default="/projects/metis0_ssd/users/vijayv/SciREX/scirex_dataset/release_data/",
                            help="If set, we will only extract citation contexts from a limited number of documents.")
    parser.add_argument('--new_scirex_data_directory',
                            type=str,
                            default="/projects/metis0_ssd/users/vijayv/SciREX/scirex_dataset/data_with_citances/", 
                            help="If set, we will only extract citation contexts from a limited number of documents.")
    args = parser.parse_args()

    # Load citation graph
    scirex_paper_ids = list(get_scirex_docids())
    scirex_to_s2orc_id_mapping = get_scirex_to_s2orc_mappings()
    scirex_s2orc_ids = set([scirex_to_s2orc_id_mapping[p] for p in scirex_paper_ids if p in scirex_to_s2orc_id_mapping])

    _, in_citations = get_citation_graph(1)
    citing_paper_ids = list([node for neighbor_set in in_citations.values() for node in neighbor_set])

    _, scigraph_documents_path = get_scirex_neighbor_texts()
    document_reader = load_full_text_documents(scigraph_documents_path, num_documents_to_load=args.num_documents_to_load)

    citation_contexts = construct_citation_contexts(document_reader, citing_paper_ids, scirex_s2orc_ids, context_window_size=2)

    #from IPython import embed; embed(); raise ValueError("breakpoint 2")

    os.makedirs(args.new_scirex_data_directory, exist_ok=True)
    shutil.copyfile(os.path.join(args.input_scirex_data_directory, "dev.jsonl"),
                    os.path.join(args.new_scirex_data_directory, "dev.jsonl"))
    shutil.copyfile(os.path.join(args.input_scirex_data_directory, "test.jsonl"),
                    os.path.join(args.new_scirex_data_directory, "test.jsonl"))
    # Should we run this for test, and validation too?
    augment_rows_with_citation_context(
                    os.path.join(args.input_scirex_data_directory, "train.jsonl"),
                    os.path.join(args.new_scirex_data_directory, "train.jsonl"),
                    citation_contexts,
                    scirex_to_s2orc_id_mapping)    

    end = time.perf_counter()
    print(f"Script took {end - start} seconds to run.")


    

if __name__ == "__main__":
    main()