import argparse
from collections import defaultdict
import json
import os

start_tokens = ["{+{", "<+<", "{-{", "<-<"]
end_tokens = ["}+}", ">+>", "}-}", ">->"]
token_meaning = ["tp", "fn", "tn", "fp"]
start_to_end_mapping = dict(zip(start_tokens, end_tokens))
token_to_meaning_mapping = dict(zip(start_tokens, token_meaning))


def sanitize_error_file(error_file_contents):
    special_characters = ["{+{", "<+<", "{-{", "<-<", "}+}", ">+>", "}-}", ">->"]
    for w in special_characters:
        error_file_contents = error_file_contents.replace(w, "")
    return error_file_contents

def load_skip_start(file):
    lines = open(file).read().split("\n")
    words = "\n".join(lines[4:])
    return words

def extract_span_types(file_contents):
    tokens = file_contents.split()
    token_types = defaultdict(list)
    clean_tokens = []
    for i, t in enumerate(tokens):
        sanitized_token = sanitize_error_file(t)
        clean_tokens.append(sanitized_token)
        outer_loop_breaker = False
        for start_token in start_tokens:
            if outer_loop_breaker:
                break
            if start_token in t:
                for ji, end_token in enumerate(tokens[i:]):
                    j = ji + i
                    if start_to_end_mapping[start_token] in end_token:
                        token_type = token_to_meaning_mapping[start_token]
                        token_types[token_type].append((i, j))
                        outer_loop_breaker = True
                        break
    
    return token_types, clean_tokens


def compute_changes(token_types_base, token_types_diff, words):
    change_spans = defaultdict(list)
    for start_type in token_meaning:
        for span in token_types_base[start_type]:
            for end_type in token_meaning:
                if end_type == start_type:
                    continue
                if span in token_types_diff[end_type]:
                    change_spans[f"{start_type}->{end_type}"].append(words[span[0]:span[1]+1])
    return change_spans



def compare_files(file_name, base_directory, diff_directory):
    base_file = os.path.join(base_directory, file_name)
    diff_file = os.path.join(diff_directory, file_name)
    token_types_base, all_tokens_1 = extract_span_types(load_skip_start(base_file))
    token_types_diff, all_tokens_2 = extract_span_types(load_skip_start(diff_file))
    assert set(all_tokens_1).issubset(all_tokens_2)
    assert "citance" not in base_file
    changes = compute_changes(token_types_base, token_types_diff, all_tokens_1)
    return changes

def pprint(ddict):
    print("{")
    for k, v in ddict.items():
        entries = ['\t\t"'+" ".join(phrase) + '"' for phrase in v]
        entries = "\n".join(entries)
        print(f"\t{k}:\t{len(v)}\n{entries},")
    print("}")


def process_files(file_names, input_directory, output_directory):
    for f in file_names:
        input_contents = open(os.path.join(input_directory, f)).read()
        output_contents = sanitize_error_file(input_contents)
        with open(os.path.join(output_directory, f), 'w') as outfile:
            outfile.write(output_contents)
    print(f"Processed {len(file_names)} files")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_directory',
                            type=str,
                            default=None,
                            help="If set, we will only extract citation contexts from a limited number of documents.")
    parser.add_argument('--diff_directory',
                            type=str,
                            default=None,
                            help="If set, we will only extract citation contexts from a limited number of documents.")
    args = parser.parse_args()

    os.makedirs(args.output_directory)
    file_names = os.listdir(args.input_directory)
    if len(file_names) == 0:
        print(f"No file names found in directory f{args.input_directory}!")
        breakpoint()
    process_files(file_names, args.input_directory, args.output_directory)


    pprint(compare_files("3", "/home/vijayv/error_files/salient_mentions_error_files", "/home/vijayv/error_files/salient_mentions_error_files_with_deepwalk"))
    pprint(compare_files("3", "/home/vijayv/error_files/salient_mentions_error_files", "/home/vijayv/error_files/salient_mentions_error_files_with_citances"))
    pprint(compare_files("3", "/home/vijayv/error_files/salient_mentions_error_files_with_deepwalk", "/home/vijayv/error_files/salient_mentions_error_files_with_graph_and_citances"))

if __name__ == "__main__":
    main()