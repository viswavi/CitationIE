# python convert_sem_eval_to_SciREX.py
import jsonlines
import os

from scienceie2017_scripts.util import readAnn
from tokenizer import make_tok_seg

# Construct spaCy sentence segmenter.
nlp = make_tok_seg()
RELATION_NAMES = ['Hyponym-of', 'Synonym-of']

SEMEVAL_DIRECTORY = "/projects/ogma1/vijayv/ScigraphIE/Sem_Eval_preparation"
TRAIN_DIRECTORY = os.path.join(SEMEVAL_DIRECTORY, "scienceie2017_train/train2")
DEV_DIRECTORY   = os.path.join(SEMEVAL_DIRECTORY, "dev")
TEST_DIRECTORY  = os.path.join(SEMEVAL_DIRECTORY, "semeval_articles_test")

def sentence_and_word_indices_from_text(text):
    sentence_indices = []
    word_indices = []
    words = []
    prev_sentence_end_idx = 0
    for sent in nlp(text).sents:
        for word in sent:
            start_idx = word.idx
            end_idx = start_idx + len(word)
            word_indices.append([start_idx, end_idx])
            words.append(word.text)
        new_sentence_end_idx = prev_sentence_end_idx + len(sent)
        sentence_indices.append([prev_sentence_end_idx, new_sentence_end_idx])
        prev_sentence_end_idx = new_sentence_end_idx
    return sentence_indices, word_indices, words


def match_char_index_to_words(char_idx, word_indices, start_word=False):
    for i, (word_start, word_end) in enumerate(word_indices):
        if (start_word and char_idx >= word_start) or (not start_word and char_idx > word_start):
            if (not start_word and char_idx <= word_end) or (start_word and char_idx < word_end):
                return i
    return -1

def convert_single_document_to_scirex(doc_id, semeval_doc):
    reformatted = {} # Keys: coref, coref_non_salient, doc_id, method_subrelations, n_ary_relations, ner, sections, sentences, words
    # TODO:  n_ary_relations, ner
    # DONE:  coref, doc_id, sections, sentences, words
    # EMPTY: coref_non_salient, method_subrelations

    document_text = semeval_doc[0]
    annotations = semeval_doc[1]


    sentence_indices, word_indices, words = sentence_and_word_indices_from_text(document_text)
    reformatted["words"] = words
    reformatted["sentences"] = sentence_indices
    reformatted["sections"] = [[0, len(words)]]
    reformatted["doc_id"] = doc_id

    # Empty/unused fields
    coref_non_salient = []
    reformatted["coref_non_salient"] = coref_non_salient
    method_subrelations = []
    reformatted["method_subrelations"] = method_subrelations

    # Coreference clusters (each word goes to separate cluster)
    corefs = {}
    ner = []
    relations = []
    entity_code_mappings = {}
    # NER
    for doc_idx, annotation in enumerate(annotations):
        if annotation["type"] == "entity":
            char_start = int(annotation['start'])
            char_end = int(annotation['end'])
            entity_type = annotation['keytype']
            entity_code = annotation['annotation_line_code']

            start_word = match_char_index_to_words(char_start, word_indices, start_word=True)
            end_word = match_char_index_to_words(char_end, word_indices, start_word=False)
            if start_word == -1 or end_word == -1:
                breakpoint()
            ner.append([start_word, end_word + 1, entity_type])
            # TODO: figure out what we need to add in here
            entity_code_mappings[entity_code] = None

            entity_phrase = " ".join(words[start_word:end_word+1])
            word_key = f"{entity_phrase}_{entity_code}"
            if word_key in corefs:
                marker = "Already seen"
                breakpoint()
            corefs[word_key] = [[start_word, end_word + 1]]
            entity_code_mappings[entity_code] = word_key

    # Relations
    for annotation in annotations:
        if annotation['type'] != "relation":
            continue
        relation_type = annotation["relation_type"]
        relation_args = annotation["relation_args"]
        if relation_type == "Hyponym-of":
            assert len(relation_args) == 2, breakpoint()
            relation = {
                "Arg1": relation_args[0],
                "Arg2": relation_args[1],
                "Type": relation_type,
            }
            relations.append(relation)
        elif relation_type == "Synonym-of":
            assert len(relation_args) >= 2
            for i in range(len(relation_args)-1):
                relation = {
                    "Arg1": relation_args[i],
                    "Arg2": relation_args[i+1],
                    "Type": relation_type,
                }
                relations.append(relation)
        else:
            breakpoint()
            raise ValueError("!")

    reformatted["coref"] = corefs
    reformatted["ner"] = ner
    reformatted["n_ary_relations"] = relations

    return reformatted

def convert_docs_to_scirex(semeval_documents, outfile):
    out = jsonlines.open(outfile, 'w')
    for filename, semeval_doc in semeval_documents.items():
        doc_id = filename.split('.ann')[0]
        scirex_doc = convert_single_document_to_scirex(doc_id, semeval_doc)
        out.write(scirex_doc)
    out.close()
    print(f"Wrote document to {outfile}.")

if __name__ == "__main__":
    train_annotations = readAnn(TRAIN_DIRECTORY)
    test_annotations = readAnn(TEST_DIRECTORY)
    dev_annotations = readAnn(DEV_DIRECTORY)



    train_out = os.path.join(SEMEVAL_DIRECTORY, "SciREX_format", "train.jsonl")
    test_out = os.path.join(SEMEVAL_DIRECTORY, "SciREX_format", "test.jsonl")
    dev_out = os.path.join(SEMEVAL_DIRECTORY, "SciREX_format", "dev.jsonl")

    convert_docs_to_scirex(train_annotations, train_out)
    convert_docs_to_scirex(test_annotations, test_out)
    convert_docs_to_scirex(dev_annotations, dev_out)